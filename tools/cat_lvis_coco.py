import json
import copy
from collections import defaultdict

COCO_PATH = 'datasets/coco/annotations/instances_train2017.json'
LVIS_PATH = 'datasets/lvis/lvis_v0.5_train.json'
SAVE_PATH = 'datasets/coco_lvis/0_lvis_v0.5_coco1232_train.json'
COCO2LVIS_PATH = 'tools/coco2lvis_id.py'

# This mapping is extracted from the official LVIS mapping:
# https://github.com/lvis-dataset/lvis-api/blob/master/data/coco_to_synset.json
COCO_SYNSET_CATEGORIES = [
    {"synset": "person.n.01", "coco_cat_id": 1},
    {"synset": "bicycle.n.01", "coco_cat_id": 2},
    {"synset": "car.n.01", "coco_cat_id": 3},
    {"synset": "motorcycle.n.01", "coco_cat_id": 4},
    {"synset": "airplane.n.01", "coco_cat_id": 5},
    {"synset": "bus.n.01", "coco_cat_id": 6},
    {"synset": "train.n.01", "coco_cat_id": 7},
    {"synset": "truck.n.01", "coco_cat_id": 8},
    {"synset": "boat.n.01", "coco_cat_id": 9},
    {"synset": "traffic_light.n.01", "coco_cat_id": 10},
    {"synset": "fireplug.n.01", "coco_cat_id": 11},
    {"synset": "stop_sign.n.01", "coco_cat_id": 13},
    {"synset": "parking_meter.n.01", "coco_cat_id": 14},
    {"synset": "bench.n.01", "coco_cat_id": 15},
    {"synset": "bird.n.01", "coco_cat_id": 16},
    {"synset": "cat.n.01", "coco_cat_id": 17},
    {"synset": "dog.n.01", "coco_cat_id": 18},
    {"synset": "horse.n.01", "coco_cat_id": 19},
    {"synset": "sheep.n.01", "coco_cat_id": 20},
    {"synset": "beef.n.01", "coco_cat_id": 21},
    {"synset": "elephant.n.01", "coco_cat_id": 22},
    {"synset": "bear.n.01", "coco_cat_id": 23},
    {"synset": "zebra.n.01", "coco_cat_id": 24},
    {"synset": "giraffe.n.01", "coco_cat_id": 25},
    {"synset": "backpack.n.01", "coco_cat_id": 27},
    {"synset": "umbrella.n.01", "coco_cat_id": 28},
    {"synset": "bag.n.04", "coco_cat_id": 31},
    {"synset": "necktie.n.01", "coco_cat_id": 32},
    {"synset": "bag.n.06", "coco_cat_id": 33},
    {"synset": "frisbee.n.01", "coco_cat_id": 34},
    {"synset": "ski.n.01", "coco_cat_id": 35},
    {"synset": "snowboard.n.01", "coco_cat_id": 36},
    {"synset": "ball.n.06", "coco_cat_id": 37},
    {"synset": "kite.n.03", "coco_cat_id": 38},
    {"synset": "baseball_bat.n.01", "coco_cat_id": 39},
    {"synset": "baseball_glove.n.01", "coco_cat_id": 40},
    {"synset": "skateboard.n.01", "coco_cat_id": 41},
    {"synset": "surfboard.n.01", "coco_cat_id": 42},
    {"synset": "tennis_racket.n.01", "coco_cat_id": 43},
    {"synset": "bottle.n.01", "coco_cat_id": 44},
    {"synset": "wineglass.n.01", "coco_cat_id": 46},
    {"synset": "cup.n.01", "coco_cat_id": 47},
    {"synset": "fork.n.01", "coco_cat_id": 48},
    {"synset": "knife.n.01", "coco_cat_id": 49},
    {"synset": "spoon.n.01", "coco_cat_id": 50},
    {"synset": "bowl.n.03", "coco_cat_id": 51},
    {"synset": "banana.n.02", "coco_cat_id": 52},
    {"synset": "apple.n.01", "coco_cat_id": 53},
    {"synset": "sandwich.n.01", "coco_cat_id": 54},
    {"synset": "orange.n.01", "coco_cat_id": 55},
    {"synset": "broccoli.n.01", "coco_cat_id": 56},
    {"synset": "carrot.n.01", "coco_cat_id": 57},
    # {"synset": "frank.n.02", "coco_cat_id": 58},
    {"synset": "sausage.n.01", "coco_cat_id": 58},
    {"synset": "pizza.n.01", "coco_cat_id": 59},
    {"synset": "doughnut.n.02", "coco_cat_id": 60},
    {"synset": "cake.n.03", "coco_cat_id": 61},
    {"synset": "chair.n.01", "coco_cat_id": 62},
    {"synset": "sofa.n.01", "coco_cat_id": 63},
    {"synset": "pot.n.04", "coco_cat_id": 64},
    {"synset": "bed.n.01", "coco_cat_id": 65},
    {"synset": "dining_table.n.01", "coco_cat_id": 67},
    {"synset": "toilet.n.02", "coco_cat_id": 70},
    {"synset": "television_receiver.n.01", "coco_cat_id": 72},
    {"synset": "laptop.n.01", "coco_cat_id": 73},
    {"synset": "mouse.n.04", "coco_cat_id": 74},
    {"synset": "remote_control.n.01", "coco_cat_id": 75},
    {"synset": "computer_keyboard.n.01", "coco_cat_id": 76},
    {"synset": "cellular_telephone.n.01", "coco_cat_id": 77},
    {"synset": "microwave.n.02", "coco_cat_id": 78},
    {"synset": "oven.n.01", "coco_cat_id": 79},
    {"synset": "toaster.n.02", "coco_cat_id": 80},
    {"synset": "sink.n.01", "coco_cat_id": 81},
    {"synset": "electric_refrigerator.n.01", "coco_cat_id": 82},
    {"synset": "book.n.01", "coco_cat_id": 84},
    {"synset": "clock.n.01", "coco_cat_id": 85},
    {"synset": "vase.n.01", "coco_cat_id": 86},
    {"synset": "scissors.n.01", "coco_cat_id": 87},
    {"synset": "teddy.n.01", "coco_cat_id": 88},
    {"synset": "hand_blower.n.01", "coco_cat_id": 89},
    {"synset": "toothbrush.n.01", "coco_cat_id": 90},
]


def cat_int(a):
    a = str(a)+str(111111)
    return(int(a))


def main():
    coco_json= json.load(open(COCO_PATH, 'r'))
    lvis_json= json.load(open(LVIS_PATH, 'r'))
    
    anns = []
    for ann in lvis_json['annotations']:
        ann['image_id'] = cat_int(ann['image_id'])
        anns.append(ann)
    
    images = []
    for image in lvis_json['images']:
        image['id'] = cat_int(image['id'])
        images.append(image) 
    
    _ = lvis_json.pop('annotations')
    _ = lvis_json.pop('images')
    lvis_json['annotations'] = anns
    lvis_json['images'] = images
        
    
    file_name_key = 'file_name' if 'v0.5' in LVIS_PATH else 'coco_url'

    lvis_cats = lvis_json['categories']

    coco2lviscats = {}
    synset2lvisid = {x['synset']: x['id'] for x in lvis_cats}
    # cocoid2synset = {x['coco_cat_id']: x['synset'] for x in COCO_SYNSET_CATEGORIES}
    coco2lviscats = {x['coco_cat_id']: synset2lvisid[x['synset']] \
        for x in COCO_SYNSET_CATEGORIES if x['synset'] in synset2lvisid}
    
    lvis_file2id = {x[file_name_key][-16:]: x['id'] for x in lvis_json['images']}

    coco2lvis = {}
    coco_id2img = {x['id']: x for x in coco_json['images']}
    coco_catid2cat = {x['id']: x for x in coco_json['categories']}
    for ann in coco_json['annotations']:
        coco_img = coco_id2img[ann['image_id']]
        file_name = coco_img['file_name'][-16:]
        # if ann['category_id'] in coco2lviscats and file_name in lvis_file2id:
        if ann['category_id'] in coco2lviscats:
            lvis_cat_id = coco2lviscats[ann['category_id']]
            # save the coco->lvis cat id
            if (ann['category_id']  not in coco2lvis):
                coco2lvis.update({ann['category_id']:lvis_cat_id})
            # print('coco_id:{} --> lvis_id{}'.format(ann['category_id'], lvis_cat_id))
            ann['category_id'] = lvis_cat_id

        # elif(ann['category_id'] not in coco2lvis):
        else:
            cocoid = ann['category_id']
            ann['category_id'] = 1+len(lvis_json['categories'])
            # save the coco->lvis cat id
            coco2lvis.update({cocoid : ann['category_id']})
            coco2lviscats.update({cocoid : ann['category_id']})
                
            cat = coco_catid2cat[cocoid]
            cat['id'] = ann['category_id'] 
            cat['synonyms'] = [cat['name']]
            # lvis_file2id.update({file_name : ann['category_id'] })
            # add to cats
            lvis_json['categories'].append(cat)
                
        # add to ann
        ann['id'] = len(lvis_json['annotations'])+1
        lvis_json['annotations'].append(ann)
        
    for img in coco_json['images']:
        lvis_json['images'].append(img)

    # print("************start saving************")
    # with open(SAVE_PATH, 'w') as f:
    #     json.dump(lvis_json, f)
    # with open(COCO2LVIS_PATH, 'w') as f:
    #     json.dump(coco2lvis, f)
    # print("************saving done************")


if __name__ == "__main__":
    main()    























