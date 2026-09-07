from util import visualize_segmentation_comparison
from dataset.loaders import PennFudanDataset
from dataset.config import PENN_F
from pipeline import run_yolo_unet_pipeline
import torch
from config import CHECKPOINT_DIR
import segmentation_models_pytorch as smp

CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/best.pt"

# coco_dataset = COCOPersonDataset(COCO_PERSON["root_dir"])
pennfudan_dataset = PennFudanDataset(PENN_F["root_dir"])


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
)
model.to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Loaded fine-tuned weights.")



sample1 = pennfudan_dataset[0]  
sample2 = pennfudan_dataset[51]  
test_samples = [
    {'image': sample1['image'], 'mask': sample1['mask'], 'name': "PennFudan Sample 1"},
    {'image': sample2['image'], 'mask': sample2['mask'], 'name': "PennFudan Sample 2"}
]

# Visualization 
visualize_segmentation_comparison(model, run_yolo_unet_pipeline, test_samples, save_path="results/penn_fudan.png")

sample1 = pennfudan_dataset[0]  
sample2 = pennfudan_dataset[50]  
test_samples = [
    {'image': sample1['image'], 'mask': sample1['mask'], 'name': "PennFudan Sample 1"},
    {'image': sample2['image'], 'mask': sample2['mask'], 'name': "PennFudan Sample 2"}
]

# Visualization 
visualize_segmentation_comparison(model, run_yolo_unet_pipeline, test_samples, save_path="results/penn-fudan.png")
