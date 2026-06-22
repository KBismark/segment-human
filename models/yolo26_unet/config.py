import os
from datasets.config import CHECKPOINT_DIR

YOLO26_UNET_CHECKPOINT_DIR = f"{CHECKPOINT_DIR}/yolo26_unet/checkpoints"
os.makedirs(YOLO26_UNET_CHECKPOINT_DIR, exist_ok=True)

