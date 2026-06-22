import time
import numpy as np
import pandas as pd
from PIL import Image


def compute_metrics(pred_mask, gt_mask):
    """
    Computes binary segmentation metrics for a single image.
    pred_mask, gt_mask: 2D numpy arrays of 0s and 1s, same shape.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

    union = tp + fp + fn
    iou = tp / union if union > 0 else 1.0 

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "iou": iou,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }


def resize_mask_to_match(mask, target_shape):
    """Resize a predicted mask to match ground truth dimensions."""
    if mask.shape == target_shape:
        return mask
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.resize((target_shape[1], target_shape[0]), Image.NEAREST)
    return (np.array(mask_img) > 127).astype(np.uint8)


def evaluate_pipeline_on_dataset(model, pipeline_fn, dataset, model_name, dataset_name, conf_threshold=0.4, max_samples=None):
    """
    Runs a segmentation pipeline across a full dataset and 
    computes metrics per image, then aggregates.

    pipeline_fn: function(image_pil) -> (predicted_mask, num_detections)
    dataset: a Dataset object returning {"image", "mask", "filename"}

    Returns: (results_df, summary_dict)
    """
    records = []
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    for idx in range(n):
        sample = dataset[idx]
        image = sample["image"]
        gt_mask = sample["mask"]

        start_time = time.time()
        pred_mask, num_detections = pipeline_fn(image, model, conf_threshold)
        inference_time = time.time() - start_time

        pred_mask = resize_mask_to_match(pred_mask, gt_mask.shape)
        metrics = compute_metrics(pred_mask, gt_mask)

        records.append({
            "model": model_name,
            "dataset": dataset_name,
            "filename": sample["filename"],
            "num_detections": num_detections,
            "iou": metrics["iou"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "inference_time_sec": inference_time
        })

        if (idx + 1) % 50 == 0:
            print(f"  [{model_name} / {dataset_name}] Processed {idx + 1}/{n} images")

    results_df = pd.DataFrame(records)

    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "num_images": n,
        "mean_iou": results_df["iou"].mean(),
        "mean_accuracy": results_df["accuracy"].mean(),
        "mean_precision": results_df["precision"].mean(),
        "mean_recall": results_df["recall"].mean(),
        "mean_inference_time_sec": results_df["inference_time_sec"].mean(),
        "fps": 1.0 / results_df["inference_time_sec"].mean() if results_df["inference_time_sec"].mean() > 0 else 0
    }

    return results_df, summary
    
