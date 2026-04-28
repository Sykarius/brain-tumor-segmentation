import os
from torch.utils.data import DataLoader
from monai.apps import DecathlonDataset

from .transforms import get_dual_pipeline_transforms
from utils.config import DataConfig

def get_dataloaders(
    config: DataConfig
):
    os.makedirs(config.root_dir, exist_ok=True)

    train_ds = DecathlonDataset(
        root_dir=config.root_dir,
        task="Task01_BrainTumour",
        section="training",
        transform=get_dual_pipeline_transforms(train=True),
        download=True,
        cache_num=config.cache_num,
    )

    val_ds = DecathlonDataset(
        root_dir=config.root_dir,
        task="Task01_BrainTumour",
        section="validation",
        transform=get_dual_pipeline_transforms(
            train=False, drop_index=config.drop_index, prob=config.prob
        ),
        download=False,
        cache_num=config.cache_num,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=config.num_workers, pin_memory=True
    )

    return train_loader, val_loader
