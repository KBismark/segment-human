import random
from torch.utils.data import Subset

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
