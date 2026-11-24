"""SinGAN re-implementation and discriminator-free Sliced Wasserstein variant.

Modules:
    singan.models    -- per-scale generator / discriminator, frozen Inception-v3 feature extractor.
    singan.swd       -- differentiable Sliced Wasserstein distance used as a training loss.
    singan.sampling  -- load a trained pyramid of generators and draw samples.
    singan.metrics   -- Wasserstein, Gaussian W2 and LPIPS helpers used to evaluate samples.

Usage:
    from singan.sampling import load_model, build_pyramid, generate_multiscale
    generators, sigma_n, n_scales, r = load_model("models/oiseau/basic", device="cpu")
    pyramid = build_pyramid("images/oiseau.png", n_scales, r)
    sample = generate_multiscale(pyramid, generators, sigma_n, n_scales, "cpu")[0]
"""
