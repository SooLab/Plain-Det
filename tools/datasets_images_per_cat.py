from pycocotools.coco import COCO
import json
ANN = '/public/home/zhuyuchen530/LLM_Det/v3/detrex/datasets/ODinW/PascalVOC/train/annotations_without_background.json'
coco = COCO(ANN)

img_ids = coco.getImgIds()
vg_catinfo = []

cat_ids = coco.getCatIds()
cats = coco.loadCats(cat_ids)
cat_names = [cat['name'] for cat in cats]

cat_img_count = {cat_id: 0 for cat_id in cat_ids}

for img_id in img_ids:
    img_ann_ids = coco.getAnnIds(imgIds=img_id)
    img_anns = coco.loadAnns(img_ann_ids)
    img_cat_ids = set([ann['category_id'] for ann in img_anns])
    for cat_id in img_cat_ids:
        cat_img_count[cat_id] += 1

for cat_id, cat_name in zip(cat_ids, cat_names):
    temp = {
        'name':cat_name,
        'image_count':cat_img_count[cat_id],
        'id':cat_id
    }
    vg_catinfo.append(temp)
    print(f"{cat_name}: {cat_img_count[cat_id]} images")

# json.dump(vg_catinfo,open('/public/home/zhuyuchen530/projects/detrex/datasets/metadata/vg_train_cat_info.py','w'))