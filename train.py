from finetune import fine_tune_unet_with_validation
from pipeline import unet_model, DEVICE
from util import stratified_split
from dataset.loaders import COCOPersonDataset
from dataset.config import COCO_PERSON
from config import CHECKPOINT_DIR
import random
import json
import torch
from torch.utils.data import Subset, ConcatDataset

def run_training():
    coco_train, coco_val = stratified_split(COCOPersonDataset(COCO_PERSON["root_dir"]))

    # train
    finetuned_model, training_history = fine_tune_unet_with_validation(
        combined_train, combined_val, unet_model, DEVICE, conf_threshold=0.4, num_epochs=80, 
        val_every=5, checpoint_dir=CHECKPOINT_DIR
    )

    save_path = f"{CHECKPOINT_DIR}/final.pt"
    torch.save(finetuned_model.state_dict(), save_path)
    print(f"Saved fine-tuned weights to {save_path}")


    with open(f"{CHECKPOINT_DIR}/training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)


if __name__ == "__main__":
    run_training()
