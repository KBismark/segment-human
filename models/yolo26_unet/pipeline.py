import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import segmentation_models_pytorch as smp
import torchvision.transforms as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# For detection-only checkpoint (not -seg), 
# since we use it purely for region proposal
yolo_model = YOLO("yolo26s.pt")
yolo_model.to(DEVICE)


# smp's "pretrained" weights only cover the ENCODER (ImageNet backbone).
# The DECODER is randomly initialized and has never learned to segment anything.
# Running it as-is produces near-random masks.
unet_model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,            # binary: person vs background
    activation=None        # raw logits; sigmoid applied manually at inference
)
unet_model.to(DEVICE)

unet_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def detect_person_boxes(image_pil, conf_threshold=0.4):
    """
    Use YOLO26 detection to return bounding boxes for 'person' class only.
    """
    results = yolo_model.predict(image_pil, conf=conf_threshold, verbose=False)
    boxes = []
    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            if int(cls) == 0:  # person class
                boxes.append(box.cpu().numpy().astype(int))
    return boxes


def segment_crop_with_unet(image_pil, box):
    """
    Crop the detected region and run UNET to get a binary mask.
    Returns the mask resized back to the crop's original dimensions.
    """
    x1, y1, x2, y2 = box
    crop = image_pil.crop((x1, y1, x2, y2))
    orig_w, orig_h = crop.size

    if orig_w == 0 or orig_h == 0:
        return None

    input_tensor = unet_transform(crop).unsqueeze(0).to(DEVICE)

    unet_model.eval()
    with torch.no_grad():
        logits = unet_model(input_tensor)
        pred = torch.sigmoid(logits).squeeze().cpu().numpy()

    binary_pred = (pred > 0.5).astype(np.uint8)
    mask_resized = np.array(
        Image.fromarray(binary_pred * 255).resize((orig_w, orig_h))
    )
    mask_resized = (mask_resized > 127).astype(np.uint8)

    return mask_resized


def run_yolo_unet_pipeline(image_pil, conf_threshold=0.4):
    """
    Full pipeline: detect persons with YOLO26, segment each crop with UNET,
    and reassemble into a full-image binary mask.
    """
    full_mask = np.zeros((image_pil.height, image_pil.width), dtype=np.uint8)

    boxes = detect_person_boxes(image_pil, conf_threshold)

    for box in boxes:
        x1, y1, x2, y2 = box
        crop_mask = segment_crop_with_unet(image_pil, box)
        if crop_mask is not None:
            full_mask[y1:y2, x1:x2] = np.maximum(
                full_mask[y1:y2, x1:x2], crop_mask
            )

    return full_mask, len(boxes)

