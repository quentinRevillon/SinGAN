"""Draw random samples from a pretrained SinGAN or SWD model.

Usage (from the repository root):
    python sample.py --model models/oiseau/basic --image images/oiseau.png --num-samples 8 --out-dir outputs/oiseau_basic

Arguments:
    --model        model folder containing generators.pt, sigma_n.json and Description.txt
    --image        training image of that model (only its pyramid sizes are used)
    --num-samples  number of images to generate (default 8)
    --out-dir      output folder, created if needed; files are named 1.png, 2.png, ...
    --seed         random seed (default 0)
"""

import argparse
import os

import torch

from singan.sampling import build_pyramid, generate_multiscale, load_model, pil_from_minus1to1


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    """Load the model, generate `num_samples` images at the finest scale and save them as PNG."""
    args = parse_args()
    torch.manual_seed(args.seed)
    generators, sigma_n, n_scales, r = load_model(args.model, device="cpu")
    pyramid = build_pyramid(args.image, n_scales, r)
    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(1, args.num_samples + 1):
        sample = generate_multiscale(pyramid, generators, sigma_n, n_scales, "cpu")[0]
        pil_from_minus1to1(sample).save(os.path.join(args.out_dir, f"{i}.png"))
    print(f"Saved {args.num_samples} samples to {args.out_dir}")


if __name__ == "__main__":
    main()
