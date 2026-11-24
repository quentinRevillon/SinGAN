"""Networks used at each scale of the SinGAN pyramid.

Usage:
    from singan.models import Generator, Discriminator, InceptionV3_Multi, weights_init
    G = Generator(n_hidden=32)
    G.apply(weights_init)
    feats = InceptionV3_Multi(device="cpu")(images)  # list of 3 feature maps
"""

import torch.nn as nn
from torchvision import models


def conv_block(in_channels, out_channels):
    """Return the layers of a size-preserving 3x3 conv, batch norm, LeakyReLU(0.2) block.

    A list (not a Sequential) keeps state_dict keys flat, as in the saved checkpoints.
    """
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    ]


class Generator(nn.Module):
    """Fully convolutional generator of one scale: 4 conv blocks then a 3x3 conv to RGB with tanh.

    The residual connection of SinGAN (x_n = x_up + G(z + x_up)) is applied by the caller.
    """

    def __init__(self, n_hidden):
        """Build the generator with `n_hidden` channels in every hidden layer."""
        super().__init__()
        self.n_hidden = n_hidden
        self.main = nn.Sequential(
            *conv_block(3, n_hidden),
            *conv_block(n_hidden, n_hidden),
            *conv_block(n_hidden, n_hidden),
            *conv_block(n_hidden, n_hidden),
            nn.Conv2d(n_hidden, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        """Map a (B, 3, H, W) tensor in [-1, 1] to a (B, 3, H, W) residual in [-1, 1]."""
        return self.main(x)


class Discriminator(nn.Module):
    """Patch discriminator (WGAN-GP critic) of one scale: 4 conv blocks then a 3x3 conv to one map."""

    def __init__(self, n_hidden):
        """Build the critic with `n_hidden` channels in every hidden layer."""
        super().__init__()
        self.n_hidden = n_hidden
        self.features = nn.Sequential(
            *conv_block(3, n_hidden),
            *conv_block(n_hidden, n_hidden),
            *conv_block(n_hidden, n_hidden),
            *conv_block(n_hidden, n_hidden),
        )
        self.output_layer = nn.Conv2d(n_hidden, 1, kernel_size=3, padding=1)

    def forward(self, x, return_features=False):
        """Return the (B, 1, H, W) patch score map, or the last hidden features if requested."""
        features = self.features(x)
        if return_features:
            return features
        return self.output_layer(features)


class InceptionV3_Multi(nn.Module):
    """Frozen ImageNet Inception-v3 returning features at three depths.

    Mixed_5d (35x35x288, fine details), Mixed_6e (17x17x768, mid-level structure),
    Mixed_7c (8x8x2048, global content), for a 299x299 ImageNet-normalised input.
    """

    LAYERS = ("Mixed_5d", "Mixed_6e", "Mixed_7c")

    def __init__(self, device="cpu"):
        """Load pretrained weights, freeze them and split the network into three blocks."""
        super().__init__()
        net = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
        net.eval()
        for p in net.parameters():
            p.requires_grad = False

        self.blocks = nn.ModuleDict()
        layers = []
        for name, module in net.named_children():
            if name in {"AuxLogits", "fc"}:
                continue
            layers.append(module)
            if name in self.LAYERS:
                self.blocks[name] = nn.Sequential(*layers)
                layers = []
            if name == self.LAYERS[-1]:
                break
        self.device = device
        self.to(device)

    def forward(self, img, layers=LAYERS):
        """Return the list of feature maps for `layers`, in the same order."""
        outs = []
        h = img
        for name in layers:
            h = self.blocks[name](h)
            outs.append(h)
        return outs


def weights_init(m):
    """DCGAN initialisation: conv weights ~ N(0, 0.02), norm weights ~ N(1, 0.02), biases at 0."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif "BatchNorm" in classname or "InstanceNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
