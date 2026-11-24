"""Differentiable Sliced Wasserstein distance (SWD) between point clouds, used as a training loss.

Usage:
    from singan.swd import extract_patches_batched, preprocess_features, sliced_wasserstein_distance
    loss = sliced_wasserstein_distance(preprocess_features(real_feats), preprocess_features(fake_feats))
    pixel_loss = sliced_wasserstein_distance(extract_patches_batched(real, 10, 10),
                                             extract_patches_batched(fake, 10, 10))
"""

import torch
import torch.nn.functional as F


def extract_patches_batched(imgs, patch_size, stride):
    """Return all sliding patches of a (B, C, H, W) batch as a (B * L, C * patch_size**2) tensor."""
    patches = F.unfold(imgs, kernel_size=patch_size, stride=stride)  # (B, C*k*k, L)
    return patches.transpose(1, 2).reshape(-1, patches.shape[1])


def preprocess_features(features):
    """Turn a (B, C, H, W) feature map into a cloud of B * H * W points of dimension C."""
    b, c, h, w = features.shape
    return features.permute(0, 2, 3, 1).reshape(b * h * w, c)


def sliced_wasserstein_distance(real_points, fake_points, num_projections=128):
    """Monte-Carlo SW1 distance between two (N, D) point clouds with the same number of points.

    Both clouds are projected on `num_projections` random unit directions; the 1D Wasserstein
    distance on each direction is the mean absolute gap between sorted projections.
    """
    projections = torch.randn(real_points.shape[1], num_projections, device=real_points.device)
    projections = projections / projections.norm(dim=0, keepdim=True)
    real_sorted, _ = torch.sort(real_points @ projections, dim=0)
    fake_sorted, _ = torch.sort(fake_points @ projections, dim=0)
    return torch.mean(torch.abs(real_sorted - fake_sorted))
