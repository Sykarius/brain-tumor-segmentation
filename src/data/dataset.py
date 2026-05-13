import os
from torch.utils.data import DataLoader
from monai.apps import DecathlonDataset
from monai.data import Dataset

from .transforms import get_dual_pipeline_transforms
from utils.config import DataConfig


def get_dataloaders(
    config: DataConfig
):
    
    if config.test:
        return _get_test_dataloaders(config)

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


def _get_data_dicts(data_path, repeat_test_samples):
    data_dicts = []

    images_dir = os.path.join(data_path, "images")
    labels_dir = os.path.join(data_path, "labels")
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Ensure both 'images' and 'labels' directories exist in {data_path}")

    # Grab all .nii.gz files from the images directory
    sample_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
    
    for filename in sample_files:
        image_path = os.path.join(images_dir, filename)
        label_path = os.path.join(labels_dir, filename)
        
        # Safety check: ensure the corresponding label exists
        if not os.path.exists(label_path):
            print(f"Warning: Label file missing for {filename}. Skipping this sample.")
            continue

        # "image" is now a single string path to the 4D file, not a list of 4 paths
        base_dict = {"image": image_path, "label": label_path}
        data_dicts.append(base_dict)

    if not data_dicts:
        raise ValueError(f"No valid image/label pairs found in {data_path}")

    data_dicts = data_dicts * repeat_test_samples
    return data_dicts


def _get_test_dataloaders(config: DataConfig):

    if not os.path.exists(config.root_dir):
        raise FileNotFoundError(f"The local data path does not exist: {config.root_dir}")

    # 1. Generate the dictionaries pointing to the images/ and labels/ folders
    data_dicts = _get_data_dicts(config.root_dir, config.repeat_test_samples)

    # 2. Build the datasets using the dual pipeline transforms
    train_ds = Dataset(
        data=data_dicts, transform=get_dual_pipeline_transforms(train=True)
    )
    val_ds = Dataset(
        data=data_dicts,
        transform=get_dual_pipeline_transforms(
            train=False, drop_index=config.drop_index, prob=config.prob
        ),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=config.num_workers, 
        pin_memory=True
    )

    return train_loader, val_loader