import os
from monai.bundle import ConfigParser
from monai.transforms import Compose, MapTransform, EnsureTyped, RandomizableTransform, CopyItemsd, EnsureChannelFirstd
import torch

class ConvertMSDBrainTumourLabelsd(MapTransform):
    """
    Convert MSD Task01_BrainTumour labels to BraTS-style region masks.

    Raw MSD labels:
        0 = background
        1 = edema
        2 = non-enhancing tumor
        3 = enhancing tumor

    Output channels:
        0 = TC = labels 2 or 3
        1 = WT = labels 1, 2, or 3
        2 = ET = label 3
    """

    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            label = d[key]

            # Raw label is usually [H, W, D].
            # If a singleton channel exists, remove it.
            if label.ndim == 4 and label.shape[0] == 1:
                label = label[0]

            tc = (label == 2) | (label == 3)
            wt = (label == 1) | (label == 2) | (label == 3)
            et = label == 3

            d[key] = torch.stack([tc, wt, et], dim=0).float()

        return d

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
        MapTransform.__init__(self, keys, allow_missing_keys)
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
    config_path = os.path.join(bundle_dir, "configs", "train.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Bundle config not found at {config_path}.")

    parser = ConfigParser()
    parser.read_config(config_path)
    
    official_transforms = parser.get_parsed_content("train#preprocessing")
    return official_transforms


def get_dual_pipeline_transforms(
    train: bool, 
    modalities: int = 4, 
    prob: float = 0.8, 
    drop_index: int = 2
):
    official_transforms = get_bundle_transforms()
    transform_list = list(official_transforms.transforms)

    transform_list.insert(1, EnsureChannelFirstd(keys=["image"], channel_dim=-1))

    # MSD Task01 labels use 1/2/3, while MONAI's built-in BraTS converter
    # expects the older 1/2/4 convention. Replace only the label converter.
    for i, transform in enumerate(transform_list):
        if transform.__class__.__name__ == "ConvertToMultiChannelBasedOnBratsClassesd":
            transform_list[i] = ConvertMSDBrainTumourLabelsd(keys=["label"])
            break

    # Duplicate the preprocessed image before dropping anything
    transform_list.append(CopyItemsd(keys=["image"], times=1, names=["image_full"]))
    
    if train:
        transform_list.append(
            RandomDropModalityd(keys=["image"], modalities=modalities, prob=prob)
        )
    else:
        transform_list.append(
            DropModalityd(keys=["image"], modalities=modalities, drop_index=drop_index)
        )
    
    transform_list.append(EnsureTyped(keys=["image", "image_full", "label"], dtype=torch.float32))
    
    return Compose(transform_list)
