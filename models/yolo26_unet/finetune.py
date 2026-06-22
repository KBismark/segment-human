from models.yolo26_unet.evaluate import compute_metrics, resize_mask_to_match
from models.yolo26_unet.pipeline import unet_transform, run_yolo_unet_pipeline
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

# Training hyperparameters 
NUM_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
TRAIN_SUBSET_SIZE = 300
CHECKPOINT_DIR = f"{os.getcwd()}/models/yolo26_unet/pretrained_weights"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)



def crop_to_bbox_from_mask(mask):
    """Get bounding box from a binary mask"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


class UNetTrainingDataset(torch.utils.data.Dataset):
    """
    Wraps dataset loaders and produces (cropped_image, cropped_mask) pairs 
    using ground-truth boxes for decoder fine-tuning.
    """
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
        self.mask_resize = T.Resize((256, 256), interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        img = sample["image"]
        mask = sample["mask"]

        bbox = crop_to_bbox_from_mask(mask)
        if bbox is None:
            # If no person in this sample; return a zero-mask crop of the full image
            crop_img = img
            crop_mask = np.zeros((img.height, img.width), dtype=np.uint8)
        else:
            x1, y1, x2, y2 = bbox
            crop_img = img.crop((x1, y1, x2, y2))
            crop_mask = mask[y1:y2, x1:x2]

        img_tensor = self.transform(crop_img)

        mask_pil = Image.fromarray(crop_mask * 255)
        mask_pil = self.mask_resize(mask_pil)
        mask_tensor = torch.from_numpy(np.array(mask_pil) > 127).float().unsqueeze(0)

        return img_tensor, mask_tensor


def dice_loss(pred_logits, target, eps=1e-6):
    pred = torch.sigmoid(pred_logits)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def fine_tune_unet_with_validation(base_train_dataset, val_dataset, model, device, num_epochs=8, val_every=1, checpoint_dir="./models/yolo26_unet"):
    """
    Fine-tune UNet with validation. Saves the best weights.
    """
    train_ds = UNetTrainingDataset(base_train_dataset, unet_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    criterion_bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_iou = 0.0
    best_epoch = 0
    history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion_bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}: train loss = {avg_train_loss:.4f}")

        if (epoch + 1) % val_every == 0:
            model.eval()
            val_ious = []
            for i in range(len(val_dataset)):
                sample = val_dataset[i]
                pred_mask, _ = run_yolo_unet_pipeline(sample["image"], 0.4)
                pred_resized = resize_mask_to_match(pred_mask, sample["mask"].shape)
                m = compute_metrics(pred_resized, sample["mask"])
                val_ious.append(m["iou"])

            val_iou = np.mean(val_ious)
            print(f"  → Validation mIoU: {val_iou:.4f}")
            history.append({"epoch": epoch+1, "train_loss": avg_train_loss, "val_iou": val_iou})

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                best_epoch = epoch + 1
                torch.save(model.state_dict(), f"{checpoint_dir}/best.pt")
                print(f"  → New best model saved (epoch {best_epoch})")

    print(f"\nBest epoch: {best_epoch}, Best validation mIoU: {best_val_iou:.4f}")
    return model, history
