from monai.transforms import (
    Compose,
    MapTransform,
    RandomizableTransform,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    NormalizeIntensityd,
    SpatialPadd,
    RandSpatialCropd,
    CenterSpatialCropd,
    ConvertToMultiChannelBasedOnBratsClassesd,
    CopyItemsd,
    EnsureTyped,
)
import torch


class DropModalityd(MapTransform):
    def __init__(self, keys, modalities: int, drop_index: int, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.modalities = modalities
        self.drop_index = drop_index

        if self.drop_index >= self.modalities:
            raise ValueError(
                f"The drop index {self.drop_index} has to be less than {self.modalities}"
            )

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # Expected shape: [C, H, W, D]
            d[key][self.drop_index, ...] = torch.zeros_like(d[key][self.drop_index, ...])
        return d


class RandomDropModalityd(RandomizableTransform, MapTransform):
    def __init__(self, keys, modalities: int, prob: float = 0.5, allow_missing_keys: bool = False):
        RandomizableTransform.__init__(self)
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.modalities = list(range(modalities))
        self.prob = prob

    def __call__(self, data):
        d = dict(data)

        if self.R.random() < self.prob:
            drop_index = self.R.choice(self.modalities)
            for key in self.keys:
                # Expected shape: [C, H, W, D]
                d[key][drop_index, ...] = torch.zeros_like(d[key][drop_index, ...])

        return d


def get_dual_pipeline_transforms(
    train: bool,
    modalities: int = 4,
    prob: float = 0.8,
    drop_index: int = 2,
    spatial_size=(96, 96, 96),
):
    """
    Lightweight BraTS/MSD transform pipeline.

    Output:
        image:      [4, H, W, D]
        image_full: [4, H, W, D]
        label:      [3, H, W, D]  # TC, WT, ET
    """

    transforms = [
        LoadImaged(keys=["image", "label"]),

        # MSD BrainTumour image loads as [H, W, D, 4].
        # Convert to PyTorch/MONAI channel-first format: [4, H, W, D].
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),

        # Label loads as [H, W, D].
        # Add channel dimension: [1, H, W, D].
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),

        Orientationd(keys=["image", "label"], axcodes="RAS"),

        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

        # Pad before cropping so small dimensions are safe.
        SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
    ]

    if train:
        transforms.append(
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=spatial_size,
                random_size=False,
            )
        )
    else:
        transforms.append(
            CenterSpatialCropd(
                keys=["image", "label"],
                roi_size=spatial_size,
            )
        )

    transforms.extend(
        [
            # Convert raw BraTS labels 1/2/4 into 3 binary channels: TC, WT, ET.
            ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),

            # Teacher gets full image; student gets dropped-modality image.
            CopyItemsd(keys=["image"], times=1, names=["image_full"]),
        ]
    )

    if train:
        transforms.append(
            RandomDropModalityd(keys=["image"], modalities=modalities, prob=prob)
        )
    else:
        transforms.append(
            DropModalityd(keys=["image"], modalities=modalities, drop_index=drop_index)
        )

    transforms.append(
        EnsureTyped(keys=["image", "image_full", "label"], dtype=torch.float32)
    )

    return Compose(transforms)
