"""Metrics comparing generated samples to the training image (report, sections 3 and 6).

Each image is mapped to a cloud of 64 Inception-v3 feature vectors (8x8 grid, 2048 dims);
clouds are compared with the exact Wasserstein distance (POT), its sliced approximation,
or the closed-form W2 distance between fitted Gaussians (SIFID-like). LPIPS is computed
with the `lpips` package on images prepared by `preprocess_image_for_lpips`.

Usage:
    from singan.metrics import load_inception, wasserstein_image, preprocess_image_for_lpips
    model = load_inception()
    d = wasserstein_image(model, "samples/oiseau/1.png", "images/oiseau.png")

    import lpips
    lpips_vgg = lpips.LPIPS(net="vgg")
    p = lpips_vgg(preprocess_image_for_lpips("samples/oiseau/1.png"),
                  preprocess_image_for_lpips("images/oiseau.png")).item()
"""

import numpy as np
import ot
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

N_POINTS = 64  # 8x8 spatial positions of the last Inception block
FEATURE_DIM = 2048


def extract_patches(img, patch_size, stride):
    """Return the sliding patches of a (C, H, W) image as a (num_patches, C * patch_size**2) tensor."""
    patches = torch.nn.functional.unfold(img.unsqueeze(0), kernel_size=patch_size, stride=stride)
    return patches.squeeze(0).T


def get_gaussian_params(points):
    """Return the empirical mean and unbiased covariance of an (N, D) point cloud."""
    mu = points.mean(dim=0)
    centered = points - mu
    return mu, centered.T @ centered / (points.shape[0] - 1)


def matrix_sqrt_eig(mat, eps=1e-10):
    """Symmetric square root of an SPD matrix by eigendecomposition (eigenvalues clamped at eps)."""
    mat = (mat + mat.T) / 2
    vals, vecs = torch.linalg.eigh(mat)
    sqrt_mat = vecs @ torch.diag(torch.sqrt(torch.clamp(vals, min=eps))) @ vecs.T
    return (sqrt_mat + sqrt_mat.T) / 2


def wasserstein_2_gaussian_eig(mu1, sigma1, mu2, sigma2, eps=1e-12):
    """Squared W2 distance between N(mu1, sigma1) and N(mu2, sigma2), divided by the dimension.

    Inputs must be float64: float32 is not precise enough for the matrix square roots.
    """
    eye = torch.eye(sigma1.size(0), device=sigma1.device, dtype=torch.float64)
    s1 = matrix_sqrt_eig(sigma1 + eps * eye, eps=eps)
    cross = matrix_sqrt_eig(s1 @ sigma2 @ s1, eps=eps)
    trace_term = torch.trace(sigma1 + sigma2 - 2 * cross)
    return (torch.norm(mu1 - mu2) ** 2 + trace_term) / len(mu1)


def load_inception():
    """Return a pretrained Inception-v3 (torch hub) whose output is the flattened Mixed_7c map."""
    model = torch.hub.load("pytorch/vision:v0.10.0", "inception_v3", pretrained=True)
    model.eval()
    model.avgpool = nn.Identity()
    model.dropout = nn.Identity()
    model.fc = nn.Identity()
    return model


def preprocess_image(file_path):
    """Load an image as a (1, 3, 299, 299) ImageNet-normalised tensor."""
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(Image.open(file_path).convert("RGB")).unsqueeze(0)


@torch.no_grad()
def inception_point_cloud(model, file_path):
    """Return the (64, 2048) cloud of Mixed_7c feature vectors of an image file."""
    return model(preprocess_image(file_path))[0].view(FEATURE_DIM, N_POINTS).T.numpy()


def wasserstein_image(model, file_path1, file_path2):
    """Exact W1 distance (Euclidean ground cost) between the feature clouds of two images."""
    a = inception_point_cloud(model, file_path1)
    b = inception_point_cloud(model, file_path2)
    weights = np.ones(N_POINTS) / N_POINTS
    return ot.emd2(weights, weights, ot.dist(a, b, metric="euclidean"))


def sliced_wasserstein_image(model, file_path1, file_path2, n_projections=100):
    """Sliced Wasserstein approximation of `wasserstein_image` with `n_projections` directions."""
    a = inception_point_cloud(model, file_path1)
    b = inception_point_cloud(model, file_path2)
    weights = np.ones(N_POINTS) / N_POINTS
    return ot.sliced_wasserstein_distance(a, b, weights, weights, n_projections)


def preprocess_image_for_lpips(file_path, image_size=256):
    """Load an image as a (1, 3, image_size, image_size) tensor in [-1, 1], as expected by LPIPS."""
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return preprocess(Image.open(file_path).convert("RGB")).unsqueeze(0)
