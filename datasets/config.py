import os

DATASETS_PATH = f"{os.getcwd()}/data"
os.makedirs(DATASETS_PATH, exist_ok=True)

LIP = {
    "dataset": "mattmdjaga/human_parsing_dataset",
    "split": "train",
    "img_dir": os.path.join(DATASETS_PATH, "LIP/image"),
    "mask_dir": os.path.join(DATASETS_PATH, "LIP/gt")
}
os.makedirs(LIP["img_dir"], exist_ok=True)
os.makedirs(LIP["mask_dir"], exist_ok=True)

COCO_PERSON = {
    "img_dir": os.path.join(DATASETS_PATH, "COCO_person/images"),
    "mask_dir": os.path.join(DATASETS_PATH, "COCO_person/masks")
}
os.makedirs(COCO_PERSON["img_dir"], exist_ok=True)
os.makedirs(COCO_PERSON["mask_dir"], exist_ok=True)

