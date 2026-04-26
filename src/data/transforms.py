import os
from monai.bundle import ConfigParser
from monai.transforms import Compose, MapTransform, EnsureTyped, RandomizableTransform, CopyItemsd
import torch

class DropModalityd(MapTransform):
    def __init__(self, keys, modalities: int, drop_index: int, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.modalities = modalities
        self.drop_index = drop_index

        if self.drop_index >= self.modalities:
            raise ValueError(f"The drop index {self.drop_index} has to be less than {self.modalities}")

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key][self.drop_index, ...] = torch.zeros_like(d[key][self.drop_index, ...])
        return d
    
class RandomDropModalityd(RandomizableTransform, MapTransform):
    def __init__(self, keys, modalities: int, prob: float = 0.5, allow_missing_keys: bool = False):
        MapTransform.__init__(keys, allow_missing_keys)
        self.modalities = list(range(modalities))
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        
        if self.R.random() < self.prob:
            drop_index = self.R.choice(self.modalities)
            for key in self.keys:
                d[key][drop_index, ...] = torch.zeros_like(d[key][drop_index, ...])   
        return d

def get_bundle_transforms(bundle_dir: str = "./bundles/brats_mri_segmentation"):
    config_path = os.path.join(bundle_dir, "configs", "inference.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Bundle config not found at {config_path}.")

    parser = ConfigParser()
    parser.read_config(config_path)
    
    official_transforms = parser.get_parsed_content("preprocessing")
    return official_transforms


def get_dual_pipeline_transforms(
    train: bool, 
    modalities: int = 4, 
    prob: float = 0.8, 
    drop_index: int = 2
):
    official_transforms = get_bundle_transforms()
    transform_list = list(official_transforms.transforms)
    
    # Duplicate the preprocessed image before dropping anything
    transform_list.append(CopyItemsd(keys=["image"], times=1, names=["image_full"]))
    
    if train:
        transform_list.append(
            RandomDropModalityd(keys=["image"], modalities=modalities, prob=prob)
        )
    else:
        transform_list.append(
            DropModalityd(keys=["image"], drop_index=drop_index)
        )
    
    transform_list.append(EnsureTyped(keys=["image", "image_full"], dtype=torch.float32))
    transform_list.append(EnsureTyped(keys=["label"], dtype=torch.long))
    
    return Compose(transform_list)