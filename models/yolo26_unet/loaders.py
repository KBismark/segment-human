import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class PennFudanDataset(Dataset):
    """
    Penn-Fudan Pedestrian Dataset.
    Structure:
        PennFudanPed/
            PNGImages/   
            PedMasks/  
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "PNGImages")
        self.mask_dir = os.path.join(root_dir, "PedMasks")
        self.imgs = sorted(os.listdir(self.img_dir))
        self.masks = sorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path))

        # Convert instance mask to binary mask 
        binary_mask = (mask > 0).astype(np.uint8)

        return {
            "image": img,
            "mask": binary_mask,
            "filename": self.imgs[idx]
        }


class LIPDataset(Dataset):
    """
    Look Into Person (LIP) Dataset subset.
    Structure:
        LIP/
            image/   
            gt/      
    """
    def __init__(self, root_dir, subset_size=None, seed=42):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "image")
        self.mask_dir = os.path.join(root_dir, "gt")

        all_imgs = sorted(os.listdir(self.img_dir))

        if subset_size and subset_size < len(all_imgs):
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(all_imgs), size=subset_size, replace=False)
            self.imgs = [all_imgs[i] for i in sorted(indices)]
        else:
            self.imgs = all_imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        mask_name = img_name.rsplit(".", 1)[0] + ".png"

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path))
        binary_mask = (mask > 0).astype(np.uint8)

        return {
            "image": img,
            "mask": binary_mask,
            "filename": img_name
        }


class COCOPersonDataset(Dataset):
    """
    COCO person-category subset.
    Structure:
        COCO_person/
            images/   
            masks/    
    """
    def __init__(self, root_dir, subset_size=None, seed=42):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")

        all_imgs = sorted(os.listdir(self.img_dir))

        if subset_size and subset_size < len(all_imgs):
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(all_imgs), size=subset_size, replace=False)
            self.imgs = [all_imgs[i] for i in sorted(indices)]
        else:
            self.imgs = all_imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        mask_name = img_name.rsplit(".", 1)[0] + ".png"

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path))
        binary_mask = (mask > 0).astype(np.uint8)

        return {
            "image": img,
            "mask": binary_mask,
            "filename": img_name
        }

