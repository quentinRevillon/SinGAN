"""Load a trained pyramid of generators and draw new samples from it.

A model folder (e.g. models/oiseau/basic) contains:
    generators.pt    list of generator state_dicts, index 0 = finest scale
    sigma_n.json     noise standard deviation per scale
    Description.txt  "key: value" lines with the training hyper-parameters (N_scales, r, ...)
    z_star.pt        fixed reconstruction noise

Usage:
    from singan.sampling import load_model, build_pyramid, generate_multiscale, pil_from_minus1to1
    generators, sigma_n, n_scales, r = load_model("models/oiseau/basic", device="cpu")
    pyramid = build_pyramid("images/oiseau.png", n_scales, r)
    images = generate_multiscale(pyramid, generators, sigma_n, n_scales, "cpu")
    pil_from_minus1to1(images[0]).save("sample.png")
"""

import json
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import resize, to_pil_image, to_tensor

from .models import Generator


def pil_from_minus1to1(t):
    """Convert a (C, H, W) or (1, C, H, W) tensor in [-1, 1] to a PIL image."""
    t = t.squeeze(0) if t.dim() == 4 else t
    return to_pil_image(((t.clamp(-1, 1) + 1) / 2).cpu())


def build_generator_from_state(sd, device="cpu"):
    """Instantiate a frozen Generator whose width is inferred from the first conv of `sd`."""
    n_hidden = next(v.shape[0] for k, v in sd.items() if k.endswith(".weight") and v.dim() == 4)
    G = Generator(n_hidden).to(device)
    G.load_state_dict(sd, strict=True)
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)
    return G


def read_description(path):
    """Parse a Description.txt file of "key: value" lines into a dict of strings."""
    with open(path, encoding="utf-8") as f:
        return dict(line.strip().split(": ", 1) for line in f if ": " in line)


def load_model(model_dir, device="cpu"):
    """Return (generators, sigma_n, n_scales, r) stored in `model_dir`."""
    state_list = torch.load(os.path.join(model_dir, "generators.pt"), map_location="cpu")
    generators = [build_generator_from_state(sd, device) for sd in state_list]
    with open(os.path.join(model_dir, "sigma_n.json"), encoding="utf-8") as f:
        sigma_n = json.load(f)
    params = read_description(os.path.join(model_dir, "Description.txt"))
    return generators, sigma_n, int(params["N_scales"]), float(params["r"])


def build_pyramid(image_path, n_scales, r, max_dim=250):
    """Return the training pyramid of an image in [-1, 1], index 0 = finest (longest side = max_dim).

    The pyramid has n_scales + 1 levels, each obtained by dividing the previous size by r.
    """
    im = to_tensor(Image.open(image_path).convert("RGB"))
    _, h, w = im.shape
    scale = max_dim / max(h, w)
    levels = [resize(im.unsqueeze(0), size=[int(h * scale), int(w * scale)], antialias=True) * 2 - 1]
    for _ in range(n_scales):
        _, _, h, w = levels[-1].shape
        levels.append(resize(levels[-1], size=[max(int(h / r), 1), max(int(w / r), 1)], antialias=True))
    return levels


@torch.no_grad()
def generate_multiscale(imgs_ref, generators, sigma_n, n_scales, device, start_scale=None):
    """Generate one sample by running the pyramid from `start_scale` down to scale 0.

    start_scale = n_scales - 1 (default) draws noise at every scale (random sample);
    a smaller start_scale starts from the upsampled real image at start_scale + 1 and only
    renews the noise at finer scales. Returns the images of every scale, index 0 = finest.
    """
    if start_scale is None:
        start_scale = n_scales - 1
    if start_scale == n_scales - 1:
        prev = torch.zeros((1, 3, *imgs_ref[start_scale].shape[2:]), device=device)
    else:
        prev = imgs_ref[start_scale + 1].to(device)

    outputs = []
    for k in range(start_scale, -1, -1):
        prev = F.interpolate(prev, size=imgs_ref[k].shape[2:], mode="bilinear", align_corners=False)
        z = torch.randn_like(prev) * sigma_n[k]
        prev = generators[k](z + prev) + prev
        outputs.append(prev)
    return outputs[::-1]
