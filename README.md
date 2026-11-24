# SinGAN

PyTorch re-implementation of SinGAN [1] and a discriminator-free variant trained with a Sliced Wasserstein distance (SWD). Team project for the IMA206 course at Télécom Paris, May–June 2025.

<img height="450" src="figures/comparison.png" />

*Left to right: training image, official implementation, our re-implementation, our SWD variant.*

## Results

Mean distance between the training image and 50 generated samples (lower is better). Source: report, Tables 5–9.

| Image | Wasserstein, official | Wasserstein, ours | Wasserstein, ours SWD | LPIPS, official | LPIPS, ours | LPIPS, ours SWD |
|---|---|---|---|---|---|---|
| Grass | 26.47 | 26.77 | 28.27 | 0.323 | 0.332 | 0.482 |
| Birds | 26.05 | 24.53 | 28.07 | 0.373 | 0.296 | 0.412 |
| London | 33.91 | 33.97 | 33.10 | 0.640 | 0.646 | 0.647 |
| Zebra | 13.68 | 12.16 | 16.19 | 0.472 | 0.368 | 0.605 |

Our re-implementation is on par with the official one. The SWD variant keeps Wasserstein distances of the same order but raises LPIPS: samples follow the texture statistics but lose global structure. With the same number of scales, it trains in about 40% less time than the adversarial version.

## Quick start

```bash
pip install -r requirements.txt
python sample.py --model models/zebre/basic --image images/zebre.png --num-samples 8 --out-dir generated/zebre_basic
```

Use `models/zebre/SWD` for the SWD variant. To train on a new image, run `notebooks/01_singan_training.ipynb` (SinGAN) or `notebooks/02_swd_training.ipynb` (SWD).

## Method

- SinGAN: a pyramid of fully convolutional generators and patch critics, trained coarse-to-fine with WGAN-GP and a reconstruction loss. Written from the paper, without the official code.
- SWD variant: we replace the critic at each scale by a frozen Inception-v3. Features from three layers are matched to the training image with a differentiable SWD. The multi-scale scheme and the reconstruction loss are unchanged.
- Evaluation: Wasserstein distance between Inception feature clouds, LPIPS (VGG) [2] and t-SNE, over 50 samples per image.

Details are in the report (in French): [`docs/report.pdf`](docs/report.pdf).

## Structure

```
singan/      models, SWD, sampling, metrics
notebooks/   training, sampling and evaluation
models/      pretrained weights (basic/ and SWD/ per image)
images/      training images
samples/     generated samples
docs/        report
```

## References

1. T. R. Shaham, T. Dekel, T. Michaeli. *SinGAN: Learning a Generative Model from a Single Natural Image*. ICCV 2019. [arXiv:1905.01164](https://arxiv.org/abs/1905.01164)
2. R. Zhang et al. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*. CVPR 2018. [arXiv:1801.03924](https://arxiv.org/abs/1801.03924)

## Team

Martin De Bats, Clément Gilli, Romain Planchon, Quentin Revillon (Télécom Paris).
Supervisors: Yann Gousseau, Arthur Leclaire.
Forked from [Loudwig/SinGan](https://github.com/Loudwig/SinGan).
