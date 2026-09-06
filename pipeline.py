import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import segmentation_models_pytorch as smp
import torchvision.transforms as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

yolo_model = YOLO("yolo26s.pt").to(DEVICE)

# DECODER is randomly initialized/untrained
unet_model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)
unet_model.eval()

unet_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_yolo_unet_pipeline(image_pil, model, conf_threshold=0.4):
    
    w, h = image_pil.size
    full_mask = np.zeros((h, w), dtype=np.uint8)

    # Step 1: Detection 
    results = yolo_model.predict(image_pil, conf=conf_threshold, verbose=False)
    boxes = []
    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            if int(cls) == 0:  # person class
                boxes.append(box.cpu().numpy().astype(int))

    if not boxes:
        return full_mask, 0

    # Step 2: Cropping 
    crop_tensors = []
    valid_boxes = []
    
    for box in boxes:
        x1, y1, x2, y2 = box
        box_w, box_h = x2 - x1, y2 - y1
        
        if box_w <= 0 or box_h <= 0:
            continue
            
        crop = image_pil.crop((x1, y1, x2, y2))
        crop_tensors.append(unet_transform(crop))
        valid_boxes.append(box)

    if not crop_tensors:
        return full_mask, 0
    
    # Batch Inference 
    # Stack all crops into one batch [N, 3, 256, 256]
    batch_tensor = torch.stack(crop_tensors).to(DEVICE)

    with torch.no_grad():
        logits = model(batch_tensor)
        preds = torch.sigmoid(logits)
        masks = (preds > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)

    # Step 3: Reassembly 
    for i, box in enumerate(valid_boxes):
        x1, y1, x2, y2 = box
        box_w, box_h = x2 - x1, y2 - y1
        
        mask_pil = Image.fromarray(masks[i])
        mask_resized = np.array(
            mask_pil.resize((box_w, box_h), resample=Image.NEAREST)
        )
        
        # Merge this individual mask into the full image mask
        full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], mask_resized)

    return full_mask, len(valid_boxes) 