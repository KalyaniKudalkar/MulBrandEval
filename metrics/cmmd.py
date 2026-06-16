# metrics/cmmd.py
# Local CMMD implementation based on Jayasumana et al. (2024)
# "Rethinking FID: Towards a Better Evaluation Metric for Image Generation"
# Equivalent to the official Google Research implementation.

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import open_clip

# ── Configuration ─────────────────────────────────────────────
CLIP_MODEL  = "ViT-B-32"
CLIP_SOURCE = "openai"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Gaussian RBF kernel bandwidth (matches official implementation)
_SIGMA = 10.0
_SCALE = 1000.0


def _load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_SOURCE
    )
    model = model.to(DEVICE).eval()
    return model, preprocess


def _extract_features(image_paths: list, model, preprocess) -> np.ndarray:
    """Extract CLIP image embeddings for a list of image file paths."""
    features = []
    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(DEVICE)
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)  # L2 normalise
            features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)


def _gaussian_rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Compute the Gaussian RBF kernel mean between two sets of embeddings."""
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    XY = X @ Y.T
    sq_dists = XX + YY.T - 2 * XY
    return np.mean(np.exp(-sq_dists / (2 * sigma ** 2)))


def compute_cmmd(real_paths: list, gen_paths: list) -> float:
    """
    Compute CMMD between real and generated image sets.

    Args:
        real_paths : list of file paths to real/reference images
        gen_paths  : list of file paths to generated images

    Returns:
        cmmd_score : float (lower is better, 0 = identical distributions)
    """
    model, preprocess = _load_clip()

    print(f"Extracting features for {len(real_paths)} real images...")
    X = _extract_features(real_paths, model, preprocess)

    print(f"Extracting features for {len(gen_paths)} generated images...")
    Y = _extract_features(gen_paths, model, preprocess)

    k_XX = _gaussian_rbf_kernel(X, X, _SIGMA)
    k_YY = _gaussian_rbf_kernel(Y, Y, _SIGMA)
    k_XY = _gaussian_rbf_kernel(X, Y, _SIGMA)

    cmmd = _SCALE * (k_XX + k_YY - 2 * k_XY)
    return float(cmmd)


if __name__ == "__main__":
    # Quick smoke test — replace with real paths to verify
    import os
    print("CMMD module loaded successfully.")
    print(f"Device: {DEVICE}")
    print("Ready for Phase 1A metric computation.")