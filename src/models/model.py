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

        self.hook_configs = [
            ("down_1", 1, 32),
            ("down_2", 2, 64),
            ("down_3", 3, 128),
        ]

        self.model_features = {}
        self._register_hook()

        latent_dim = 128

        self.projector = nn.ModuleDict({
            key: self._build_projector(channels, latent_dim)
            for key, _, channels in self.hook_configs
        })
        if self.is_teacher:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
    

    
    def _build_projector(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1)
        )

    def _register_hook(self):
        def get_hook(name):
            def hook(module, input, output):
                self.model_features[name] = output
            return hook
        
        for key, idx, _ in self.hook_configs:
            self.model.down_layers[idx].register_forward_hook(get_hook(key))

    def forward(self, inputs):

        output_logits = self.model(inputs)
        
        projected_features = [
            self.projector[key](self.model_features[key])
            for key, _, _ in self.hook_configs
        ]

        return output_logits, projected_features
