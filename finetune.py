from evaluate import compute_metrics, resize_mask_to_match
from pipeline import unet_transform, run_yolo_unet_pipeline
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
NUM_EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

def crop_to_bbox_from_mask(mask):
    """Get bounding box from a binary mask"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()

class UNetTrainingDataset(torch.utils.data.Dataset):
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
            crop_img = img
            crop_mask = np.zeros((img.height, img.width), dtype=np.uint8)
        else:
            x1, y1, x2, y2 = bbox
            crop_img = img.crop((x1, y1, x2, y2))
            crop_mask = mask[y1:y2, x1:x2]

        img_tensor = self.transform(crop_img)
        mask_pil = Image.fromarray((crop_mask * 255).astype(np.uint8))
        mask_pil = self.mask_resize(mask_pil)
        mask_tensor = torch.from_numpy(np.array(mask_pil) > 127).float().unsqueeze(0)

        return img_tensor, mask_tensor

def dice_loss(pred_logits, target, eps=1e-6):
    pred = torch.sigmoid(pred_logits)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()

def fine_tune_unet_with_validation(base_train_dataset, val_dataset, model, device, conf_threshold=0.4, num_epochs=50, val_every=1, checkpoint_dir="./pretrained_weights"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    train_ds = UNetTrainingDataset(base_train_dataset, unet_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    criterion_bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    
    start_epoch = 0
    best_val_iou = 0.0
    history = []
    last_checkpoint_path = os.path.join(checkpoint_dir, "last.pt")

    if os.path.exists(last_checkpoint_path):
        print(f" -- Found last checkpoint: {last_checkpoint_path}. Resuming...")
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_iou = checkpoint['best_val_iou']
        history = checkpoint.get('history', [])
        print(f" -- Resumed from epoch {start_epoch}. Previous Best mIoU: {best_val_iou:.4f}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion_bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}: train loss = {avg_train_loss:.4f}")

        # VALIDATION 
        val_iou = 0.0
        if (epoch + 1) % val_every == 0:
            model.eval()
            val_ious = []
            val_limit = min(len(val_dataset), 200) 
            
            with torch.no_grad():
                for i in range(val_limit):
                    sample = val_dataset[i]
                    pred_mask, _ = run_yolo_unet_pipeline(sample["image"], model, conf_threshold=conf_threshold)
                    pred_resized = resize_mask_to_match(pred_mask, sample["mask"].shape)
                    m = compute_metrics(pred_resized, sample["mask"])
                    val_ious.append(m["iou"])

            val_iou = np.mean(val_ious)
            print(f" -- Validation mIoU: {val_iou:.4f}")
        
        # Save History
        history.append({"epoch": epoch+1, "train_loss": avg_train_loss, "val_iou": val_iou})

        # --- SAVE CHECKPOINTS ---
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_iou': best_val_iou,
            'history': history
        }

        # Save 'last.pt' 
        torch.save(checkpoint_data, last_checkpoint_path)

        # Save 'best.pt' if mIoU improves
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save(checkpoint_data, best_path)
            print(f" -- New best model saved: {best_path}")

    return model, history
