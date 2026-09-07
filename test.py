from evaluate import evaluate_pipeline_on_dataset
from pipeline import run_yolo_unet_pipeline
from train import run_training
import os
import json
import torch
from torch.utils.data import Subset, ConcatDataset
from dataset.loaders import COCOPersonDataset, PennFudanDataset
from dataset.config import COCO_PERSON, PENN_F
from config import CHECKPOINT_DIR, OUTPUT_DIR
import pandas as pd
import segmentation_models_pytorch as smp

CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/best.pt"

# coco_dataset = COCOPersonDataset(COCO_PERSON["root_dir"])
pennfudan_dataset = PennFudanDataset(PENN_F["root_dir"])


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

test_model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
)
test_model.to(DEVICE)

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    test_model.load_state_dict(checkpoint['model_state_dict'])
    test_model.eval()
    print("Loaded fine-tuned weights. Running tests...")
else:
    print("No fine-tuned checkpoint found. Running training script..." )
    run_training()
    print('Training complete!')
    print('Running tests...')


all_results = []
all_summaries = []

eval_sets = {
    "PennFudan": pennfudan_dataset
}

for dataset_name, ds in eval_sets.items():
    print(f"\nEvaluating YOLO26+UNET on {dataset_name} ({len(ds)} images)...")
    results_df, summary = evaluate_pipeline_on_dataset(
        test_model,
        pipeline_fn=run_yolo_unet_pipeline,
        dataset=ds,
        model_name="YOLO26+UNET",
        dataset_name=dataset_name,
        conf_threshold=0.4
    )
    # all_results.append(results_df)
    all_summaries.append(summary)
    print(f"  mIoU: {summary['mean_iou']:.4f} | Accuracy: {summary['mean_accuracy']:.4f} | "
          f"Precision: {summary['mean_precision']:.4f} | Recall: {summary['mean_recall']:.4f} | "
          f"FPS: {summary['fps']:.2f}")

# save results
# full_results_df = pd.concat(all_results, ignore_index=True)
summary_df = pd.DataFrame(all_summaries)

# full_results_df.to_csv(f"{OUTPUT_DIR}/yolo26_unet_per_image_results.csv", index=False)
summary_df.to_csv(f"{OUTPUT_DIR}/yolo26_unet_summary.csv", index=False)

print(summary_df.to_string(index=False))
print(f"\nResults saved to {OUTPUT_DIR}")

