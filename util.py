import random
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np

random.seed(42)

def stratified_split(dataset, train_frac=0.80, val_frac=0.20):
    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:]

    return (Subset(dataset, train_idx),Subset(dataset, val_idx))


import matplotlib.pyplot as plt
import numpy as np

def visualize_segmentation_comparison(model, pipeline_fn, samples, conf_threshold=0.4, save_path="prediction_results.png"):
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, sample in enumerate(samples):
        img_pil = sample['image']
        gt_mask = sample['mask']
        name = sample.get('name', f"Sample {i+1}")

        # Run Inference
        pred_mask, num_det = pipeline_fn(img_pil, model, conf_threshold)

        # Plotting
        axes[i, 0].imshow(img_pil)
        axes[i, 0].set_title(f"Image: {name}\n(Detections: {num_det})")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_mask, cmap='gray') 
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis('off')

    plt.tight_layout()
    
    # Save plot image
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Plot saved successfully to: {save_path}")
    
    plt.close(fig)
    
