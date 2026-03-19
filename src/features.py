""" Features.py - Feature extraction pipeline. """

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Image utilities
# ══════════════════════════════════════════════════════════════════════════════

def rows_to_images(X):
    """(N, 3072) -> (N, 32, 32, 3) float32."""
    # reshape flat vectors into RGB images
    N = X.shape[0]
    imgs = np.empty((N, 32, 32, 3), dtype=np.float32)
    imgs[:, :, :, 0] = X[:, :1024].reshape(N, 32, 32)
    imgs[:, :, :, 1] = X[:, 1024:2048].reshape(N, 32, 32)
    imgs[:, :, :, 2] = X[:, 2048:].reshape(N, 32, 32)
    return imgs


def to_grayscale(imgs):
    """(N,32,32,3) -> (N,32,32)."""
    # standard RGB -> grayscale conversion
    return (0.2989 * imgs[:, :, :, 0]
            + 0.5870 * imgs[:, :, :, 1]
            + 0.1140 * imgs[:, :, :, 2])


def rgb_to_hsv_batch(imgs):
    """
    (N,32,32,3) RGB -> (N,32,32,3) HSV, all channels in [0,1].
    """
    # normalise each image separately (helps with illumination differences)
    lo = imgs.reshape(imgs.shape[0], -1).min(axis=1)[:, None, None, None]
    hi = imgs.reshape(imgs.shape[0], -1).max(axis=1)[:, None, None, None]
    imgs_n = (imgs - lo) / (hi - lo + 1e-8)

    R = imgs_n[:, :, :, 0]
    G = imgs_n[:, :, :, 1]
    B = imgs_n[:, :, :, 2]

    Cmax  = np.maximum(np.maximum(R, G), B)
    Cmin  = np.minimum(np.minimum(R, G), B)
    delta = Cmax - Cmin

    # compute hue (piecewise)
    H = np.zeros_like(R)
    eps = 1e-8
    mr = (Cmax == R) & (delta > eps)
    mg = (Cmax == G) & (delta > eps)
    mb = (Cmax == B) & (delta > eps)
    H[mr] = ((G - B)[mr] / (delta[mr] + eps)) % 6
    H[mg] = ((B - R)[mg] / (delta[mg] + eps)) + 2
    H[mb] = ((R - G)[mb] / (delta[mb] + eps)) + 4
    H /= 6.0

    # saturation + value
    S = np.where(Cmax > eps, delta / (Cmax + eps), 0.0)
    V = Cmax

    return np.stack([H, S, V], axis=-1).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# HOG
# ══════════════════════════════════════════════════════════════════════════════

def _hog_single(gray, cell=4, n_bins=9, block=2):
    """HOG descriptor for one (H,W) grayscale image."""
    # simple gradient (no fancy padding)
    H, W = gray.shape
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

    mag = np.sqrt(gx*gx + gy*gy)
    ang = (np.arctan2(gy, gx) * (180.0 / np.pi)) % 180.0

    nch = H // cell
    ncw = W // cell
    hist = np.zeros((nch, ncw, n_bins), dtype=np.float32)
    bw = 180.0 / n_bins

    # build histogram per cell
    for ci in range(nch):
        for cj in range(ncw):
            m = mag[ci*cell:(ci+1)*cell, cj*cell:(cj+1)*cell].ravel()
            a = ang[ci*cell:(ci+1)*cell, cj*cell:(cj+1)*cell].ravel()
            b0 = (a / bw).astype(np.int32) % n_bins
            b1 = (b0 + 1) % n_bins
            frac = (a / bw) - b0.astype(np.float32)
            np.add.at(hist[ci, cj], b0, m * (1.0 - frac))
            np.add.at(hist[ci, cj], b1, m * frac)

    # block normalisation
    desc = []
    for bi in range(nch - block + 1):
        for bj in range(ncw - block + 1):
            v = hist[bi:bi+block, bj:bj+block, :].ravel()
            v = v / (np.linalg.norm(v) + 1e-6)
            v = np.clip(v, 0, 0.2)
            v = v / (np.linalg.norm(v) + 1e-6)
            desc.append(v)

    return np.concatenate(desc).astype(np.float32)


def compute_hog(grays, cell=4, n_bins=9, block=2):
    """HOG for all images. Returns (N, D)."""
    print(f"[HOG] cell={cell} on {len(grays)} images...")
    return np.stack([_hog_single(g, cell, n_bins, block) for g in grays])


# ══════════════════════════════════════════════════════════════════════════════
# Opponent colour HOG
# ══════════════════════════════════════════════════════════════════════════════

def _opponent_hog_single(img, cell=4, n_bins=9, block=2):
    """
    HOG using opponent colour channels.
    """
    # convert RGB -> opponent channels
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    channels = [
        (R - G) / 1.4142,
        (R + G - 2*B) / 2.4495,
        (R + G + B) / 1.7321,
    ]

    best_mag = np.zeros_like(R)
    best_ang = np.zeros_like(R)

    # pick strongest gradient among channels
    for ch in channels:
        gx = np.zeros_like(ch)
        gy = np.zeros_like(ch)
        gx[:, 1:-1] = ch[:, 2:] - ch[:, :-2]
        gy[1:-1, :] = ch[2:, :] - ch[:-2, :]
        mag = np.sqrt(gx*gx + gy*gy)
        ang = (np.arctan2(gy, gx) * (180.0 / np.pi)) % 180.0
        mask = mag > best_mag
        best_mag[mask] = mag[mask]
        best_ang[mask] = ang[mask]

    # rest is same as standard HOG
    H, W = best_mag.shape
    nch = H // cell
    ncw = W // cell
    hist = np.zeros((nch, ncw, n_bins), dtype=np.float32)
    bw = 180.0 / n_bins

    for ci in range(nch):
        for cj in range(ncw):
            m = best_mag[ci*cell:(ci+1)*cell, cj*cell:(cj+1)*cell].ravel()
            a = best_ang[ci*cell:(ci+1)*cell, cj*cell:(cj+1)*cell].ravel()
            b0 = (a / bw).astype(np.int32) % n_bins
            b1 = (b0 + 1) % n_bins
            frac = (a / bw) - b0.astype(np.float32)
            np.add.at(hist[ci, cj], b0, m * (1.0 - frac))
            np.add.at(hist[ci, cj], b1, m * frac)

    desc = []
    for bi in range(nch - block + 1):
        for bj in range(ncw - block + 1):
            v = hist[bi:bi+block, bj:bj+block, :].ravel()
            v = v / (np.linalg.norm(v) + 1e-6)
            v = np.clip(v, 0, 0.2)
            v = v / (np.linalg.norm(v) + 1e-6)
            desc.append(v)

    return np.concatenate(desc).astype(np.float32)


def compute_opponent_hog(imgs, cell=4, n_bins=9, block=2):
    """Opponent colour HOG for all images. Returns (N, D)."""
    print(f"[Opp-HOG] cell={cell}...")
    return np.stack([_opponent_hog_single(img, cell, n_bins, block)
                     for img in imgs])


# ══════════════════════════════════════════════════════════════════════════════
# Spatial Pyramid colour histogram (HSV)
# ══════════════════════════════════════════════════════════════════════════════

def compute_spatial_pyramid_color(imgs_hsv, n_bins=16, levels=(1, 2, 4)):
    """
    Spatial pyramid pooling over HSV colour histograms.
    """
    N, H, W, _ = imgs_hsv.shape
    parts = []

    for level in levels:
        cell_h = H // level
        cell_w = W // level
        n_cells = level * level
        feat = np.zeros((N, n_cells * 3 * n_bins), dtype=np.float32)

        cell_idx = 0
        for gi in range(level):
            for gj in range(level):
                patch = imgs_hsv[:, gi*cell_h:(gi+1)*cell_h,
                                     gj*cell_w:(gj+1)*cell_w, :]
                flat  = patch.reshape(N, -1, 3)

                for c in range(3):
                    col_data = flat[:, :, c]
                    for b in range(n_bins):
                        lo = b / n_bins
                        hi = (b + 1) / n_bins
                        out_col = cell_idx * 3 * n_bins + c * n_bins + b
                        feat[:, out_col] = ((col_data >= lo) &
                                            (col_data < hi)).mean(axis=1)
                cell_idx += 1

        parts.append(feat)

    result = np.concatenate(parts, axis=1)
    print(f"[SPC] done, dim={result.shape[1]}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Local patch statistics
# ══════════════════════════════════════════════════════════════════════════════

def compute_patch_stats(grays, grid=4):
    """
    Divide image into gridxgrid patches; compute mean + std per patch.
    """
    N, H, W = grays.shape
    ph = H // grid
    pw = W // grid

    stats = []
    for gi in range(grid):
        for gj in range(grid):
            patch = grays[:, gi*ph:(gi+1)*ph, gj*pw:(gj+1)*pw].reshape(N, -1)
            stats.append(patch.mean(axis=1))
            stats.append(patch.std(axis=1))

    result = np.stack(stats, axis=1).astype(np.float32)
    print(f"[Patch stats] dim={result.shape[1]}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Spatial Pyramid LBP
# ══════════════════════════════════════════════════════════════════════════════

def compute_lbp_spatial(grays, radius=1, n_points=8, levels=(1, 2, 4)):
    """
    Uniform Local Binary Pattern with spatial pyramid pooling.
    """
    N, H, W = grays.shape
    n_bins = n_points + 2

    angles = 2.0 * np.pi * np.arange(n_points) / n_points
    dy = -radius * np.sin(angles)
    dx =  radius * np.cos(angles)

    codes = np.zeros((N, n_points, H, W), dtype=np.uint8)

    # build LBP codes
    for k in range(n_points):
        yf = np.arange(H, dtype=np.float32)[:, None] + dy[k]
        xf = np.arange(W, dtype=np.float32)[None, :] + dx[k]
        y0 = np.clip(np.floor(yf).astype(np.int32), 0, H - 2)
        x0 = np.clip(np.floor(xf).astype(np.int32), 0, W - 2)
        wy = (yf - y0).astype(np.float32)
        wx = (xf - x0).astype(np.float32)

        nbr = (grays[:, y0,   x0  ] * (1 - wy) * (1 - wx)
             + grays[:, y0+1, x0  ] * wy        * (1 - wx)
             + grays[:, y0,   x0+1] * (1 - wy)  * wx
             + grays[:, y0+1, x0+1] * wy         * wx)

        codes[:, k] = (nbr >= grays).astype(np.uint8)

    transitions = np.sum(codes != np.roll(codes, 1, axis=1), axis=1)
    popcount = codes.sum(axis=1).astype(np.int32)
    lbp_map  = np.where(transitions <= 2, popcount, n_points + 1)

    parts = []
    for level in levels:
        cell_h = H // level
        cell_w = W // level
        for gi in range(level):
            for gj in range(level):
                patch = lbp_map[:, gi*cell_h:(gi+1)*cell_h,
                                    gj*cell_w:(gj+1)*cell_w]
                flat  = patch.reshape(N, -1)

                hist  = np.zeros((N, n_bins), dtype=np.float32)
                for b in range(n_bins):
                    hist[:, b] = (flat == b).sum(axis=1)

                hist /= (hist.sum(axis=1, keepdims=True) + 1e-8)
                parts.append(hist)

    result = np.concatenate(parts, axis=1).astype(np.float32)
    print(f"[LBP] done, dim={result.shape[1]}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(X,
                     cell=4, n_bins_hog=9, block=2,
                     n_bins_col=16,
                     lbp_radius=1, lbp_pts=8):
    """
    Enhanced feature pipeline for (N, 3072) raw pixel matrix.
    """
    print("Starting feature extraction...")

    imgs     = rows_to_images(X)
    grays    = to_grayscale(imgs)
    imgs_hsv = rgb_to_hsv_batch(imgs)

    hog_fine   = compute_hog(grays, cell=2, n_bins=n_bins_hog, block=2)
    hog_coarse = compute_hog(grays, cell=4, n_bins=n_bins_hog, block=2)
    hog_opp    = compute_opponent_hog(imgs,  cell=4, n_bins=n_bins_hog, block=2)
    spc        = compute_spatial_pyramid_color(imgs_hsv, n_bins=n_bins_col,
                                               levels=(1, 2, 4))
    pst        = compute_patch_stats(grays, grid=4)
    lbp        = compute_lbp_spatial(grays, radius=1, n_points=8, levels=(1, 2, 4))

    feat = np.concatenate(
        [hog_fine, hog_coarse, hog_opp, spc, pst, lbp], axis=1)

    print(f"[Done] total dim = {feat.shape[1]}")
    return feat.astype(np.float32)


def standardise(Xtr, Xte):
    """Zero-mean / unit-std normalisation using training stats only."""
    mu  = Xtr.mean(axis=0)
    std = Xtr.std(axis=0) + 1e-8
    return (Xtr - mu) / std, (Xte - mu) / std, mu, std