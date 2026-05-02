import os
from torch.utils.data import DataLoader
from monai.apps import DecathlonDataset
from monai.data import Dataset

from .transforms import get_dual_pipeline_transforms
from utils.config import DataConfig


def _get_data_dicts(data_path, repeat_test_samples):
    data_dicts = []

    samples = os.listdir(data_path)

    for sample in samples:
        image_paths = [
            os.path.join(data_path, sample, f"{sample}_flair.nii.gz"),
            os.path.join(data_path, sample, f"{sample}_t1.nii.gz"),
            os.path.join(data_path, sample, f"{sample}_t1ce.nii.gz"),
            os.path.join(data_path, sample, f"{sample}_t2.nii.gz"),
        ]
        label_path = os.path.join(data_path, sample, f"{sample}_seg.nii.gz")
        data_dicts.append({"image": image_paths, "label": label_path})

    return data_dicts * repeat_test_samples


def get_dataloaders(config: DataConfig):

    if config.test:
        return _get_test_dataloaders(config)

    os.makedirs(config.root_dir, exist_ok=True)

    extracted_dataset_dir = os.path.join(config.root_dir, "Task01_BrainTumour")
    should_download = not os.path.exists(extracted_dataset_dir)

    print(f"Dataset root: {config.root_dir}")
    print(f"Extracted dataset exists: {os.path.exists(extracted_dataset_dir)}")
    print(f"download={should_download}")

    train_ds = DecathlonDataset(
        root_dir=config.root_dir,
        task="Task01_BrainTumour",
        section="training",
        transform=get_dual_pipeline_transforms(
            train=True,
            drop_index=config.drop_index,
            prob=config.prob,
        ),
        download=should_download,
        cache_num=config.cache_num,
    )

    val_ds = DecathlonDataset(
        root_dir=config.root_dir,
        task="Task01_BrainTumour",
        section="validation",
        transform=get_dual_pipeline_transforms(
            train=False,
            drop_index=config.drop_index,
            prob=config.prob,
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
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def _get_test_dataloaders(config: DataConfig):

    if not os.path.exists(config.root_dir):
        raise FileNotFoundError(f"The local data path does not exist: {config.root_dir}")

    data_dicts = _get_data_dicts(config.root_dir, config.repeat_test_samples)

    train_ds = Dataset(
        data=data_dicts,
        transform=get_dual_pipeline_transforms(
            train=True,
            drop_index=config.drop_index,
            prob=config.prob,
        ),
    )

    val_ds = Dataset(
        data=data_dicts,
        transform=get_dual_pipeline_transforms(
            train=False,
            drop_index=config.drop_index,
            prob=config.prob,
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
        pin_memory=True,
    )

    return train_loader, val_loader
