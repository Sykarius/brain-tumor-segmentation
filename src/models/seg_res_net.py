import os
import torch
from monai.bundle import ConfigParser
from torch import nn

def get_pretrained_model(bundle_dir: str = "./bundles/brats_mri_segmentation"):
    
    config_path = os.path.join(bundle_dir, "configs", "train.json")
    weights_path = os.path.join(bundle_dir, "models", "model.pt")

    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Bundle files not found at {bundle_dir}. Did you run the download command?"
        )

    parser = ConfigParser()
    parser.read_config(config_path)
    
    # The bundle defines the architecture under the "network" key
    model = parser.get_parsed_content("network")

    # Load weights into CPU first
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    return model


class SegResNet(nn.Module):

    def __init__(self, bundle_dir: str = "./bundles/brats_mri_segmentation", is_teacher=False):
        super().__init__()

        self.model = get_pretrained_model(bundle_dir)
        self.is_teacher = is_teacher
        
        if self.is_teacher:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.model_features = None
        self._register_hook()

        penultimate_channels = 32
        latent_dim = 128

        self.projector = self._build_projector(penultimate_channels, latent_dim)

    
    def _build_projector(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1)
        )

    def _register_hook(self):
        def get_features(module, input, output):
            self.model_features = output
        feature_layer = list(self.model.children())[-2]
        feature_layer.register_forward_hook(get_features)

    def forward(self, inputs):

        output_logits = self.model(inputs)
        projected_features = self.projector(self.model_features)

        return output_logits, projected_features