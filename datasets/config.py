import os

DATASETS_PATH = f"{os.getcwd()}/data"
os.makedirs(DATASETS_PATH, exist_ok=True)

LIP = {
    "dataset": "mattmdjaga/human_parsing_dataset",
    "root_dir": os.path.join(DATASETS_PATH, "LIP"),
    "img_dir": os.path.join(DATASETS_PATH, "LIP/image"),
    "mask_dir": os.path.join(DATASETS_PATH, "LIP/gt")
}
os.makedirs(LIP["root_dir"], exist_ok=True)
os.makedirs(LIP["img_dir"], exist_ok=True)
os.makedirs(LIP["mask_dir"], exist_ok=True)

COCO_PERSON = {
    "root_dir": os.path.join(DATASETS_PATH, "COCO_person"),
    "img_dir": os.path.join(DATASETS_PATH, "COCO_person/images"),
    "mask_dir": os.path.join(DATASETS_PATH, "COCO_person/masks")
}
os.makedirs(COCO_PERSON["root_dir"], exist_ok=True)
os.makedirs(COCO_PERSON["img_dir"], exist_ok=True)
os.makedirs(COCO_PERSON["mask_dir"], exist_ok=True)

MADS = {
    "root_dir": os.path.join(DATASETS_PATH, "MADS_Dataset"),
    "img_dir": os.path.join(DATASETS_PATH, "MADS_Dataset/images"),
    "mask_dir": os.path.join(DATASETS_PATH, "MADS_Dataset/masks")
}
os.makedirs(MADS["root_dir"], exist_ok=True)
os.makedirs(MADS["img_dir"], exist_ok=True)
os.makedirs(MADS["mask_dir"], exist_ok=True)

PENN_F = {
    "root_dir": os.path.join(DATASETS_PATH, "PennFudanPed"),
    "img_dir": os.path.join(DATASETS_PATH, "PennFudanPed/PNGImages"),
    "mask_dir": os.path.join(DATASETS_PATH, "PennFudanPed/PedMasks")
}
os.makedirs(PENN_F["root_dir"], exist_ok=True)
os.makedirs(PENN_F["img_dir"], exist_ok=True)
os.makedirs(PENN_F["mask_dir"], exist_ok=True)



CHECKPOINT_DIR = f"{os.getcwd()}/models"

OUTPUT_DIR = f"{os.getcwd()}/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
