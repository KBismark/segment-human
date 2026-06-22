from models.yolo26_unet.finetune import fine_tune_unet_with_validation
from models.yolo26_unet.pipeline import unet_model, DEVICE
from datasets.loaders import COCOPersonDataset, LIPDataset, MADSDataset, PennFudanDataset
from datasets.config import COCO_PERSON, MADS, LIP, PENN_F
from config import YOLO26_UNET_CHECKPOINT_DIR
import random
import json
import torch
from torch.utils.data import Subset, ConcatDataset


random.seed(42)

def stratified_split(dataset, train_frac=0.75, val_frac=0.10, test_frac=0.15):
    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (Subset(dataset, train_idx),
            Subset(dataset, val_idx),
            Subset(dataset, test_idx))


# all four datasets 
mads_dataset = MADSDataset(MADS["root_dir"])
coco_dataset = COCOPersonDataset(COCO_PERSON["root_dir"])
lip_dataset = LIPDataset(LIP["root_dir"])
pennfudan_dataset = PennFudanDataset(PENN_F["root_dir"])

# split within each dataset
mads_train, mads_val, mads_test = stratified_split(mads_dataset)
coco_train, coco_val, coco_test = stratified_split(coco_dataset)
lip_train, lip_val, lip_test = stratified_split(lip_dataset)

# 60/40 train/test split for Penn-Fudan
pf_indices = list(range(len(pennfudan_dataset)))
random.shuffle(pf_indices)
pf_train_end = int(len(pf_indices) * 0.6)
pf_train_idx = pf_indices[:pf_train_end]
pf_test_idx = pf_indices[pf_train_end:]
pf_train,  pf_test = (Subset(pennfudan_dataset, pf_train_idx), Subset(pennfudan_dataset, pf_test_idx))

# combine each split across datasets
combined_train = ConcatDataset([mads_train, coco_train, lip_train, pf_train])
combined_val = ConcatDataset([mads_val, coco_val, lip_val])
combined_test = ConcatDataset([mads_test, coco_test, lip_test, pf_test])

split_record = {
    "mads": {
        "train": mads_train.indices,
        "val": mads_val.indices,
        "test": mads_test.indices
    },
    "coco": {
        "train": coco_train.indices,
        "val": coco_val.indices,
        "test": coco_test.indices
    },
    "lip": {
        "train": lip_train.indices,
        "val": lip_val.indices,
        "test": lip_test.indices
    },
    "pennfudan": {
        "train": pf_train.indices,
        "test": pf_test.indices
    }
}

with open(f"{YOLO26_UNET_CHECKPOINT_DIR}/split_indices.json", "w") as f:
    json.dump(split_record, f, indent=2)


#  train and run validation 
finetuned_model, training_history = fine_tune_unet_with_validation(
    combined_train, combined_val, unet_model, DEVICE, num_epochs=10, 
    val_every=2, checpoint_dir=YOLO26_UNET_CHECKPOINT_DIR
)

save_path = f"{YOLO26_UNET_CHECKPOINT_DIR}/final.pt"
torch.save(finetuned_model.state_dict(), save_path)
print(f"Saved COCO-fine-tuned UNET weights to {save_path}")


with open(f"{YOLO26_UNET_CHECKPOINT_DIR}/training_history.json", "w") as f:
    json.dump(training_history, f, indent=2)
print("Saved training history (loss + validation mIoU per epoch).")

