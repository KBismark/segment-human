from config import COCO_PERSON
import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO

# Download COCO val2017 images + annotations  and unzip
# !wget -q http://images.cocodataset.org/zips/val2017.zip
# !wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# !unzip -q val2017.zip
# !unzip -q annotations_trainval2017.zip


coco = COCO("annotations/instances_val2017.json")

person_cat_id = coco.getCatIds(catNms=["person"])
img_ids = coco.getImgIds(catIds=person_cat_id)

print(f"Total images containing at least one person: {len(img_ids)}")

NUM_SAMPLES = len(img_ids) if len(img_ids) < 2000 else 2000 # 2000 since it's over 2600
random.seed(42)
random.shuffle(img_ids)
subset_ids = img_ids[:NUM_SAMPLES]

img_dir = COCO_PERSON["img_dir"]
mask_dir = COCO_PERSON["mask_dir"]

for img_id in tqdm(subset_ids):
    img_info = coco.loadImgs(img_id)[0]
    src_path = os.path.join("val2017", img_info["file_name"])
    if not os.path.exists(src_path):
        continue

    Image.open(src_path).convert("RGB").save(os.path.join(img_dir, img_info["file_name"]))

    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=person_cat_id)
    anns = coco.loadAnns(ann_ids)

    combined_mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
    for ann in anns:
        m = coco.annToMask(ann)
        combined_mask = np.maximum(combined_mask, m)

    mask_filename = img_info["file_name"].rsplit(".", 1)[0] + ".png"
    Image.fromarray(combined_mask * 255).save(os.path.join(mask_dir, mask_filename))


