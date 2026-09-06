from config import LIP
import os
from datasets import load_dataset
import numpy as np
from PIL import Image
from tqdm import tqdm


def save_lip_subset(num_samples=2000):
    ds = load_dataset(LIP['dataset'], split=LIP['train_split'][:num_samples])
    img_dir = LIP['img_dir']
    mask_dir = LIP['mask_dir']
    

    for i, item in enumerate(tqdm(ds)):
        img_name = f"lip_{i:04d}.jpg"
        item['image'].convert("RGB").save(os.path.join(img_dir, img_name))
        mask_np = np.array(item['mask'])
        binary_mask = np.where(mask_np > 0, 255, 0).astype(np.uint8)
        Image.fromarray(binary_mask).save(os.path.join(mask_dir, img_name.replace(".jpg", ".png")))

