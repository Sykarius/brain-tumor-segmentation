import torch
from dataclasses import dataclass, field
from typing import Literal, Tuple
import yaml
import os

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@dataclass
class LossConfig:
    alpha: float
    beta: float
    gamma: float
    temperature: float
    aggregation: Literal["random", "avg_pool", "class_balanced"] = "avg_pool"
    kl_temperature: float = 2.0
    max_samples: float = 4096
    class_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)

@dataclass
class CheckpointConfig:
    save_dir: str = "./checkpoints"
    filename: str = "best_student_model.pt"
    patience: int = 15
    min_delta: float = 1e-4
    mode: Literal["max", "min"] = "max"

@dataclass
class DataConfig:
    root_dir: str = "./data"
    batch_size: int = 2
    num_workers: int = 4
    drop_index: int = 2
    cache_num: int = 100
    prob: float = 0.8

    # For simple testing the pipeline with few samples
    test: bool = False
    repeat_test_samples: int = 10


@dataclass
class TrainingConfig:
    lr: float
    epochs: int
    loss: LossConfig
    checkpoint: CheckpointConfig
    data: DataConfig
    bundle_dir: str = "./bundles/brats_mri_segmentation"
    logging_dir: str = "./logs/brats"
    weight_decay: float = 1e-5
    device: torch.device = field(default_factory=get_device)



def load_config(config_path: str) -> TrainingConfig:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The config file {config_path} does not exists")
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    loss_data = data.pop("loss", {})
    checkpoint_data = data.pop("checkpoint", {})
    data_dict = data.pop("data", {})

    loss = LossConfig(**loss_data)
    checkpoint = CheckpointConfig(**checkpoint_data)
    data_config = DataConfig(**data_dict)
    config = TrainingConfig(
        loss=loss,
        checkpoint=checkpoint,
        data=data_config,
        **data
    )

    return config