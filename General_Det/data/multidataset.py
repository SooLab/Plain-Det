# Copyright (c) Facebook, Inc. and its affiliates.
# Part of the code is from https://github.com/xingyizhou/UniDet/blob/master/projects/UniDet/unidet/data/multi_dataset_dataloader.py (Apache-2.0 License)
import copy
import os
import math
import logging
import numpy as np
import operator
import torch
import torch.utils.data
import random
import json
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import _log_api_usage, log_first_n
from detectron2.utils.logger import setup_logger

from detectron2.config import configurable
from detectron2.data import samplers
from torch.utils.data.sampler import BatchSampler, Sampler
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import get_detection_dataset_dicts, build_batch_data_loader
from detectron2.data.samplers import TrainingSampler, RepeatFactorTrainingSampler
from detectron2.data.build import worker_init_reset_seed, print_instances_class_histogram
from detectron2.data.build import filter_images_with_only_crowd_annotations
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.data.build import check_metadata_consistency
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
from detectron2.utils import comm
import itertools
import math
import time
from collections import defaultdict
from typing import Optional


def _custom_train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    if 'MultiDataset' in sampler_name:
        dataset_dicts = get_detection_dataset_dicts_with_source(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
    else:
        dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    if sampler is not None:
        pass
    elif sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "MultiDatasetSampler":
        sampler = MultiDatasetSampler(
            dataset_dicts,
            dataset_ratio = cfg.DATALOADER.DATASET_RATIO,
            use_rfs = cfg.DATALOADER.USE_RFS,
            dataset_ann = cfg.DATALOADER.DATASET_ANN,
            repeat_threshold = cfg.DATALOADER.REPEAT_THRESHOLD,
        )
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset_dicts,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        'multi_dataset_grouping': cfg.DATALOADER.MULTI_DATASET_GROUPING,
        'use_diff_bs_size': cfg.DATALOADER.USE_DIFF_BS_SIZE,
        'dataset_bs': cfg.DATALOADER.DATASET_BS,
        'num_datasets': len(cfg.DATASETS.TRAIN)
    }


@configurable(from_config=_custom_train_loader_from_config)
def build_custom_train_loader(
        dataset, *, mapper, sampler, 
        total_batch_size=16,
        aspect_ratio_grouping=True, 
        num_workers=0,
        num_datasets=1,
        multi_dataset_grouping=False,
        use_diff_bs_size=False,
        dataset_bs=[]
    ):
    """
    Modified from detectron2.data.build.build_custom_train_loader, but supports
    different samplers
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    if multi_dataset_grouping:
        return build_multi_dataset_batch_data_loader(
            use_diff_bs_size,
            dataset_bs,
            dataset,
            sampler,
            total_batch_size,
            num_datasets=num_datasets,
            num_workers=num_workers,
        )
    else:
        return build_batch_data_loader(
            dataset,
            sampler,
            total_batch_size,
            aspect_ratio_grouping=aspect_ratio_grouping,
            num_workers=num_workers,
        )


def build_multi_dataset_batch_data_loader(
    use_diff_bs_size, dataset_bs,
    dataset, sampler, total_batch_size, num_datasets, num_workers=0
):
    """
    """
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )

    batch_size = total_batch_size // world_size
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        num_workers=num_workers,
        batch_sampler=None,
        collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
        worker_init_fn=worker_init_reset_seed,
    )  # yield individual mapped dict
    if use_diff_bs_size:
        return DIFFMDAspectRatioGroupedDataset(
            data_loader, dataset_bs, num_datasets)
    else:
        return MDAspectRatioGroupedDataset(
            data_loader, batch_size, num_datasets)


def get_detection_dataset_dicts_with_source(
    dataset_names, filter_empty=True, min_keypoints=0, proposal_files=None
):
    assert len(dataset_names)
    dataset_names = [dataset_names] if isinstance(dataset_names, str) else dataset_names
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)
    
    for source_id, (dataset_name, dicts) in \
        enumerate(zip(dataset_names, dataset_dicts)):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)
        for d in dicts:
            d['dataset_source'] = source_id
            d['dataset_name'] = dataset_name

        if "annotations" in dicts[0] or "text" in dicts[0]:
            try:
                class_names = MetadataCatalog.get(dataset_name).thing_classes
                check_metadata_consistency("thing_classes", dataset_name)
                print_instances_class_histogram(dicts, class_names)
            except AttributeError:  # class names are not available for this dataset
                pass

    assert proposal_files is None

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    if min_keypoints > 0 and has_instances:
        dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)

    return dataset_dicts


class MultiDatasetSampler(Sampler):
    count = 0
    def __init__(
        self, 
        dataset_dicts, 
        dataset_ratio,
        rfs,
        dataset_ann,
        output_dir=None,
        online_sample=True,
        num_workers=4,
        total_batch_size=16,
        repeat_threshold=0.001,
        seed: Optional[int] = None,
        ):
        """
        example:
        rfs = {
            "use_rfs":[False, False, False],
            "load_rfs":[False, False, True],
            "load_path":[None,None,RFS_PATH]
        },
        """
        sizes = [0 for _ in range(len(dataset_ratio))]
        for d in dataset_dicts:
            sizes[d['dataset_source']] += 1
        print('dataset sizes:', sizes)
        self.sizes = sizes
        assert len(rfs['use_rfs']) == len(rfs['load_rfs']) == len(rfs['load_path'])
        assert len(dataset_ratio) == len(sizes), \
            'length of dataset ratio {} should be equal to number if dataset {}'.format(
                len(dataset_ratio), len(sizes)
            )
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        # print("WS:",self._world_size)
        self.dataset_ids =  torch.tensor(
            [d['dataset_source'] for d in dataset_dicts], dtype=torch.long)
        self.total_batch_size = total_batch_size
        self.num_workers = num_workers
        self.output_dir = output_dir
        self.online_sample = online_sample

        # dataset_weight = [torch.ones(s) * max(sizes) / s * r / sum(dataset_ratio) \
        #     for i, (r, s) in enumerate(zip(dataset_ratio, sizes))]

        """Attention
        if use this ,train O365 and other datasets will hardly hurt the num of larger datasets.
        """
        dataset_weight = [r*torch.ones(s) for (r,s) in zip(dataset_ratio, sizes)]
        dataset_weight = torch.cat(dataset_weight)

        rfs_factors = []
        # cas_factors = []
        st = 0
        for i, s in enumerate(sizes):
            if rfs['use_rfs'][i]:
                if dataset_ann[i] == 'box':
                    rfs_func = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency
                    # cas_func = get_class_balance_factor_per_dataset
                    # rfs_func = repeat_factors_from_category_frequency
                else:
                    rfs_func = repeat_factors_from_tag_frequency
                    pass
                
                # cas_factor = cas_func(dataset_dicts[st: st + s],l=1)
                rfs_factor = rfs_func(
                    dataset_dicts[st: st + s],
                    repeat_thresh=repeat_threshold,
                    # num_images=sum(sizes)
                    )
                rfs_factor = rfs_factor * (s / rfs_factor.sum())
                # cas_factor = cas_factor * (s / cas_factor.sum())
            elif rfs['load_rfs'][i]:
                assert rfs['load_path'][i] != None, f"want to load rfs_file but path is None"
                with open(rfs['load_path'][i], 'r') as f:
                    temp = json.load(f)
                    rfs_factor = torch.as_tensor(temp['rfs_factor'])
                    # print(rfs_factor.shape)
                rfs_factor = rfs_factor * (s / rfs_factor.sum())
                # print('2:',rfs_factor.shape)

            else:
                rfs_factor = torch.ones(s)
                # cas_factor = torch.ones(s)
            rfs_factors.append(rfs_factor)
            # cas_factors.append(cas_factor)
            st = st + s
        rfs_factors = torch.cat(rfs_factors)
        # cas_factors = torch.cat(cas_factors)

        self.weights = dataset_weight * rfs_factors
        # self.weights = dataset_weight * cas_factors
        self.sample_epoch_size = len(self.weights)
        
        # hack here
        # self.weights = [dataset_weight1,dataset_weight2]
        # self.sample_epoch_size = len(self.weights[0])
    
    @classmethod
    def get_count(cls):
        return cls.count
        

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size)


    def _infinite_indices(self):
        # self.count += 1
        g = torch.Generator()
        g.manual_seed(self._seed)
        # logger = setup_logger()
        # logger.info("worldsize::"+ str(self._world_size))
        while True:
            dataset_weight = 1
            logger = setup_logger()
            if len(self.sizes) > 1:
                loss = []
                log_path = os.path.join(self.output_dir,'losslog')
                log_json= os.path.join(log_path,'loss.json')
                log_txt = os.path.join(log_path,'x_log.txt')
                x = [1 for _ in self.sizes]

                if os.path.exists(log_json) and self.online_sample:
                    with open(log_json,'r')as f:
                        loss = json.load(f)
                    # choose the dataset 
                    if len(self.sizes) == 2:
                        del loss['o365']
                        del loss['oid']
                    elif len(self.sizes) == 3:
                        del loss['oid']
                    # calculate dataset weight
                    x = self.calculateX(loss)
                    logger.info(f"online sampling weights:{x}")
                    with open(log_txt,"a")as l:
                        l.write(str(x)+"\n")
                elif self.online_sample:
                    x = [max(self.sizes)/s for s in self.sizes]
                dataset_weight = [torch.ones(s) * r \
                    for _, (r, s) in enumerate(zip(x, self.sizes))]
                dataset_weight=torch.cat(dataset_weight)    
            else:
                pass
                # print("no online")
            
            ids = torch.multinomial(
                torch.as_tensor(self.weights)*torch.as_tensor(dataset_weight), 
                self.sample_epoch_size, generator=g,
                replacement=True
            )

            # ids = torch.multinomial(
            #     torch.as_tensor(self.weights), 
            #     self.sample_epoch_size, generator=g,
            #     replacement=True
            # )
            
            # print('not done*************')
            ids = sort_ids(ids, self.sizes, self.total_batch_size, self._world_size, self.num_workers, self.output_dir)
            nums = [(self.dataset_ids[ids] == i).sum().int().item() \
                for i in range(len(self.sizes))]
            yield from ids
    
    def calculateX(self, losses):
        length = np.array(self.sizes)
        if len(length)==4:
            a = {"coco":0.0,"lvis_v1":0.0,"o365":0.0,"oid":0.0}
        elif len(length)==3:
            a = {"coco":0.0,"lvis_v1":0.0,"o365":0.0,}
        elif len(length)==2:
            a = {"coco":0.0,"lvis_v1":0.0}
        
        for name in losses.keys():
            count = 0
            for i, loss in enumerate(losses[name]):
                count += 1
                a[name] += loss['loss_bbox']
                # writer.add_scalar(f"{name}", float(loss['loss_bbox']), i+199999)
            a[name]=a[name]/count

        x = [v for _,v in a.items() if v>0]
        x = np.array(x)
        
        x/=x.min()
        # origin sampler 
        # s = 1+ np.sqrt((x-1)*(length.max().repeat(len(length))/length))

        # new sampler 
        # s = x*np.sqrt((length.max().repeat(len(length))/length))
        print(f"we change the sampler ratio:0.3 here: {str(__file__)}!!!!!!!!")
        # s = x*(length.max().repeat(len(length))/length)
        s = x*((length.max().repeat(len(length))/length)**0.7)

        return s



class MDAspectRatioGroupedDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, batch_size, num_datasets):
        """
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self._buckets = [[] for _ in range(2 * num_datasets)]

    def __iter__(self):
        for d in self.dataset:
            w, h = d["width"], d["height"]
            aspect_ratio_bucket_id = 0 if w > h else 1
            bucket_id = d['dataset_source'] * 2 + aspect_ratio_bucket_id
            bucket = self._buckets[bucket_id]
            bucket.append(d)
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]


class DIFFMDAspectRatioGroupedDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, batch_sizes, num_datasets):
        """
        """
        self.dataset = dataset
        self.batch_sizes = batch_sizes
        self._buckets = [[] for _ in range(2 * num_datasets)]

    def __iter__(self):
        for d in self.dataset:
            w, h = d["width"], d["height"]
            aspect_ratio_bucket_id = 0 if w > h else 1
            bucket_id = d['dataset_source'] * 2 + aspect_ratio_bucket_id
            bucket = self._buckets[bucket_id]
            bucket.append(d)
            if len(bucket) == self.batch_sizes[d['dataset_source']]:
                yield bucket[:]
                del bucket[:]


def repeat_factors_from_tag_frequency(dataset_dicts, repeat_thresh):
    """
    """
    category_freq = defaultdict(int)
    for dataset_dict in dataset_dicts:
        cat_ids = dataset_dict['pos_category_ids']
        for cat_id in cat_ids:
            category_freq[cat_id] += 1
    num_images = len(dataset_dicts)
    for k, v in category_freq.items():
        category_freq[k] = v / num_images

    category_rep = {
        cat_id: max(1.0, math.sqrt(repeat_thresh / cat_freq))
        for cat_id, cat_freq in category_freq.items()
    }

    rep_factors = []
    for dataset_dict in dataset_dicts:
        cat_ids = dataset_dict['pos_category_ids']
        rep_factor = max({category_rep[cat_id] for cat_id in cat_ids}, default=1.0)
        rep_factors.append(rep_factor)

    return torch.tensor(rep_factors, dtype=torch.float32)


# # sort the idx of two datasets one by one
# def merge_idx(list1,list2,step=1):
#     m_l = min(len(list1),len(list2))
#     l1 = list1[:m_l]
#     l2 = list2[:m_l]
#     output = []
    
#     if len(list1)<len(list2):
#         exc_idx = list2[m_l:]
#     else:
#         exc_idx = list1[m_l:]
    
#     for i in range(0,m_l,step):
#         if (i+step)<m_l:
#             up = i+step
#         else:
#             up = m_l
#         output+=l1[i:up]
#         output+=l2[i:up]
    
#     output += exc_idx
#     return output
"""
input : list
"""
def random_merge(input, total_batch_size, world_size, num_workers):
    # print("input[1]:",len(input[1]))
    # print("input[2]:",len(input[2]))
    # print("test**************")
    num_workers = max(num_workers,1)
    seed = 1024
    batch_size = total_batch_size // world_size
    l = 0
    for i in input:
        l += len(i)
    weight = []
    for i in input:
        weight.append(len(i)/l)
    print('weight:',weight)

    idx = []
    remove = []
    while (len(input)!=0):
        g = torch.Generator()
        g.manual_seed(seed)

        sort_data = torch.multinomial(
            torch.as_tensor(weight), world_size, generator=g,
            replacement=True
        )
        for _ in range(batch_size):
            for i in sort_data:
                if len(input[i]) == 0:
                   print(f'debug:{remove},bs:{batch_size},sortdata:{sort_data},ws:{world_size}') 
                idx.extend([input[i].pop()]) 
        for i, temp in enumerate(input):
            if len(temp) < total_batch_size:
                remove.append(i)
                # del input[i]
                # del weight[i]
        input = np.delete(input, remove)
        weight = np.delete(weight, remove)
        remove = []
        seed += 1
    
    # to adapt to the num_workers
    output = []
    st = 0
    while(st < len(idx)):
        for i in range(total_batch_size):
            # 边界情况
            if len(idx)-st < total_batch_size*num_workers:
                num_w = (len(idx)-st)//total_batch_size
            else :
                num_w = num_workers 
            for j in range(num_w):
                id = st+i+j*total_batch_size
                output.append(idx[id]) 
        st += num_workers*total_batch_size
    redundant = int(len(output)%(total_batch_size*num_workers))
    if redundant>0:
        return output[:-redundant]
    else:
        return output
"""
"""
def merge_idx3(idx1, idx2, idx3):
    """arrange the 8*GPU(batch_size = 4), two datasets, according to the num of sampled weight


    Args:
        idx1 (list): _description_
        idx2 (list): _description_

    Returns:
        list
    """
    logger = setup_logger()

    l_1 = len(idx1)
    l_2 = len(idx2)
    l_3 = len(idx3)
    logger.info(f"l1:{l_1},l2:{l_2},l3:{l_3}")

    n_1 = min(max(round(8*l_1/(l_1+l_2+l_3)),1),8-get_world_size())
    n_2 = min(max(round(8*l_2/(l_1+l_2+l_3)),1),8-get_world_size())
    n_3 = 8-(n_1+n_2)
    print("datasets1:datasets2:datasets3 =",n_1,":",n_2,":",n_3)
    idx = []

    while(len(idx1) >= n_1 or len(idx2)>=n_2 or len(idx3)>=n_3):
        if len(idx1)>=n_1:
            idx.extend([idx1.pop() for _ in range(n_1)])
        
        if len(idx2)>=n_2:
            idx.extend([idx2.pop() for _ in range(n_2)])

        if len(idx3)>=n_3:
            idx.extend([idx3.pop() for _ in range(n_3)])
        # print(f'idx1:{len(idx1)}idx2:{len(idx2)}idx3:{len(idx3)}')
    # print('merge OK')
    total_batch_size = 32
    num_workers = 4
    output = []
    st = 0
    while(st < len(idx)):
        for i in range(total_batch_size):
            # 边界情况
            if len(idx)-st < total_batch_size*num_workers:
                num_w = (len(idx)-st)//total_batch_size
            else :
                num_w = num_workers 
            for j in range(num_w):
                id = st+i+j*total_batch_size
                # print(f"{num_w}:{id}/{len(idx)}")
                output.append(idx[id]) 
                # print("output:",len(output))
        st += num_workers*total_batch_size
    return output

def merge_idx(idx1, idx2):
    """arrange the 8*GPU(batch_size = 4), two datasets, according to the num of sampled weight


    Args:
        idx1 (list): _description_
        idx2 (list): _description_

    Returns:
        list
    """
    l_1 = len(idx1)
    l_2 = len(idx2)

    n_1 = min(max(round(8*l_1/(l_1+l_2)),1),8-get_world_size())
    n_2 = 8-n_1
    print("datasets1:datasets2 =",n_1,":",n_2)
    idx = []

    while(len(idx1) >= n_1 or len(idx2)>=n_2 ):
        if len(idx1)>=n_1:
            idx.extend([idx1.pop() for _ in range(n_1)])
        
        if len(idx2)>=n_2:
            idx.extend([idx2.pop() for _ in range(n_2)])

    return idx


def sort_ids(ids, size, total_batch_size=32, world_size=8, num_workers=4, output_dir=None):
    if len(size) == 1:
        return ids
    id1, id2, id3, id4 = [],[],[],[]
    for id in ids:
        if id < size[0]:
            id1.append(id)
        elif id >=size[0] and id < size[0]+size[1]:
            id2.append(id)
        elif len(size)>2 and id >= size[0]+size[1] and id < size[0]+size[1]+size[2]:
            id3.append(id)
        else :
            id4.append(id)

    log_path = os.path.join(output_dir,'losslog')
    if os.path.exists(log_path):
        with open(os.path.join(log_path, 'x_log.txt') ,"a")as l:
            l.write(f"datasets_len:{len(id1)}:{len(id2)}:{len(id3)}:{len(id4)}"+"\n")
    # return merge_idx3(id1, id2, id3)
    return random_merge([id1,id2,id3,id4], total_batch_size, world_size, num_workers)
        

def get_class_balance_factor_per_dataset(dataset_dicts, l=1.):
    ret = []
    category_freq = defaultdict(int)
    for dataset_dict in dataset_dicts:  # For each image (without repeats)
        cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
        for cat_id in cat_ids:
            category_freq[cat_id] += 1
    for i, dataset_dict in enumerate(dataset_dicts):
        cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
        ret.append(sum(
            [1. / (category_freq[cat_id] ** l) for cat_id in cat_ids]))
    return torch.tensor(ret).float()
# def repeat_factors_from_category_frequency(dataset_dicts, repeat_thresh, num_images):
#        """
#        Compute (fractional) per-image repeat factors based on category frequency.
#        The repeat factor for an image is a function of the frequency of the rarest
#        category labeled in that image. The "frequency of category c" in [0, 1] is defined
#        as the fraction of images in the training set (without repeats) in which category c
#        appears.
#        See :paper:`lvis` (>= v2) Appendix B.2.

#        Args:
#            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
#            repeat_thresh (float): frequency threshold below which data is repeated.
#                If the frequency is half of `repeat_thresh`, the image will be
#                repeated twice.

#        Returns:
#            torch.Tensor:
#                the i-th element is the repeat factor for the dataset image at index i.
#        """
#        # 1. For each category c, compute the fraction of images that contain it: f(c)
#        category_freq = defaultdict(int)
#        for dataset_dict in dataset_dicts:  # For each image (without repeats)
#            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
#            for cat_id in cat_ids:
#                category_freq[cat_id] += 1
#     #    num_images = len(dataset_dicts)
#        for k, v in category_freq.items():
#            category_freq[k] = v / num_images

#        # 2. For each category c, compute the category-level repeat factor:
#        #    r(c) = max(1, sqrt(t / f(c)))
#        category_rep = {
#            cat_id: max(1.0, math.sqrt(repeat_thresh / cat_freq))
#            for cat_id, cat_freq in category_freq.items()
#        }

#        # 3. For each image I, compute the image-level repeat factor:
#        #    r(I) = max_{c in I} r(c)
#        rep_factors = []
#        for dataset_dict in dataset_dicts:
#            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
#            rep_factor = max({category_rep[cat_id] for cat_id in cat_ids}, default=1.0)
#            rep_factors.append(rep_factor)

#        return torch.tensor(rep_factors, dtype=torch.float32)           
    