from models.yolo26_unet.evaluate import evaluate_pipeline_on_dataset
from models.yolo26_unet.pipeline import run_yolo_unet_pipeline
import os
import json
import torch
from torch.utils.data import Subset, ConcatDataset
from datasets.loaders import COCOPersonDataset, LIPDataset, MADSDataset, PennFudanDataset
from datasets.config import COCO_PERSON, MADS, LIP, PENN_F, OUTPUT_DIR
from config import YOLO26_UNET_CHECKPOINT_DIR
import pandas as pd
import segmentation_models_pytorch as smp

# load saved split indices 
with open(f"{YOLO26_UNET_CHECKPOINT_DIR}/split_indices.json") as f:
    split_record = json.load(f)

# test subsets from saved indices 
mads_dataset = MADSDataset(MADS["root_dir"])
coco_dataset = COCOPersonDataset(COCO_PERSON["root_dir"])
lip_dataset = LIPDataset(LIP["root_dir"])
pennfudan_dataset = PennFudanDataset(PENN_F["root_dir"])

mads_test = Subset(mads_dataset, split_record["mads"]["test"])
coco_test = Subset(coco_dataset, split_record["coco"]["test"])
lip_test = Subset(lip_dataset, split_record["lip"]["test"])
pf_test = Subset(pennfudan_dataset, split_record["pennfudan"]["test"])

# combined test set 
combined_test = ConcatDataset([mads_test, coco_test, lip_test, pf_test])


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

test_model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
)
test_model.to(DEVICE)

# load the fine-tuned UNET weights
CHECKPOINT_PATH = f"{YOLO26_UNET_CHECKPOINT_DIR}/best.pt"

if os.path.exists(CHECKPOINT_PATH):
    test_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    test_model.eval()
    print("Loaded MADS-fine-tuned UNET weights — skipping fine-tuning.")
else:
    raise FileNotFoundError(
        "No fine-tuned checkpoint found. Run [train.py] before testing."
    )


all_results = []
all_summaries = []

eval_sets = {
    "PennFudan": pf_test,
    "Main": combined_test,
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
    all_results.append(results_df)
    all_summaries.append(summary)
    print(f"  mIoU: {summary['mean_iou']:.4f} | Accuracy: {summary['mean_accuracy']:.4f} | "
          f"Precision: {summary['mean_precision']:.4f} | Recall: {summary['mean_recall']:.4f} | "
          f"FPS: {summary['fps']:.2f}")

# save results
full_results_df = pd.concat(all_results, ignore_index=True)
summary_df = pd.DataFrame(all_summaries)

full_results_df.to_csv(f"{OUTPUT_DIR}/yolo26_unet_per_image_results.csv", index=False)
summary_df.to_csv(f"{OUTPUT_DIR}/yolo26_unet_summary.csv", index=False)

print(summary_df.to_string(index=False))
print(f"\nResults saved to {OUTPUT_DIR}")

