"""
start.py - KRR + additive RBF kernel + One-vs-Rest image classifier.
"""

import sys
import os
import time
import argparse
import numpy as np
import pandas as pd

# add src folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from features import extract_features, standardise
from kernel   import build_gram_matrices
from krr      import KRROneVsRest

# feature dimensions (just to split later)
HOG_DIM = 8100 + 1764 + 1764   # 11628
COL_DIM = 1008 + 32            # 1040
LBP_DIM = (1 + 4 + 16) * 10    # 210


CONFIG = {
    'data_dir'   : '.',
    'output_file': 'Yte.csv',

    # feature params
    'hog_cell'    : 4,
    'hog_bins'    : 9,
    'hog_block'   : 2,
    'col_bins'    : 16,
    'lbp_radius'  : 1,
    'lbp_pts'     : 8,

    # kernel params (scaled by feature dimension)
    'gamma_mult_hog': 1.0,
    'gamma_mult_col': 1.5,
    'gamma_mult_lbp': 3.0,

    'lam'         : 5e-4,

    'seed'        : 42,
}

# small grid for tuning (kept relatively small for speed)
SEARCH_GRID = {
    'gamma_mult_hog': [0.75, 1.0, 1.25],
    'gamma_mult_col': [1.0, 1.5, 2.0],
    'gamma_mult_lbp': [2.0, 3.0, 4.0],
    'lam'           : [1e-4, 5e-4, 1e-3],
    'n_folds'       : 5,
    'n_sub'         : 3000,
}


def load_data(data_dir):
    # load csv data (only first 3072 cols are pixels)
    Xtr = np.array(pd.read_csv(
        os.path.join(data_dir, 'Xtr.csv'),
        header=None, sep=',', usecols=range(3072)), dtype=np.float32)
    Xte = np.array(pd.read_csv(
        os.path.join(data_dir, 'Xte.csv'),
        header=None, sep=',', usecols=range(3072)), dtype=np.float32)
    Ytr = np.array(pd.read_csv(
        os.path.join(data_dir, 'Ytr.csv'),
        sep=',', usecols=[1])).squeeze().astype(int)
    return Xtr, Xte, Ytr


def build_additive_kernel(Ftr, Fte, gamma_mult_hog, gamma_mult_col,
                          gamma_mult_lbp):
    """
    Build K = K_hog + K_col + K_lbp using per-group RBF kernels.
    """
    # split features into 3 groups
    col_end = HOG_DIM + COL_DIM
    Ftr_hog = Ftr[:, :HOG_DIM];        Fte_hog = Fte[:, :HOG_DIM]
    Ftr_col = Ftr[:, HOG_DIM:col_end]; Fte_col = Fte[:, HOG_DIM:col_end]
    Ftr_lbp = Ftr[:, col_end:];        Fte_lbp = Fte[:, col_end:]

    # scale gamma by feature dimension
    gamma_hog = gamma_mult_hog / Ftr_hog.shape[1]
    gamma_col = gamma_mult_col / Ftr_col.shape[1]
    gamma_lbp = gamma_mult_lbp / Ftr_lbp.shape[1]

    print(f"HOG dim={Ftr_hog.shape[1]}, gamma={gamma_hog:.2e}")
    print(f"Color dim={Ftr_col.shape[1]}, gamma={gamma_col:.2e}")
    print(f"LBP dim={Ftr_lbp.shape[1]}, gamma={gamma_lbp:.2e}")

    # build kernels for each feature group
    K_hog_tr, K_hog_te = build_gram_matrices(Ftr_hog, Fte_hog, gamma_hog)
    K_col_tr, K_col_te = build_gram_matrices(Ftr_col, Fte_col, gamma_col)
    K_lbp_tr, K_lbp_te = build_gram_matrices(Ftr_lbp, Fte_lbp, gamma_lbp)

    return (K_hog_tr + K_col_tr + K_lbp_tr,
            K_hog_te + K_col_te + K_lbp_te)


def run_search(Xtr, Ytr, cfg):
    """Grid search over gamma_mult_hog, gamma_mult_col, gamma_mult_lbp, lam."""
    print("\nRunning hyperparameter search...")

    rng   = np.random.RandomState(cfg['seed'])
    n_sub = SEARCH_GRID['n_sub']
    idx   = rng.choice(len(Ytr), min(n_sub, len(Ytr)), replace=False)
    Xs, Ys = Xtr[idx], Ytr[idx]

    print(f"Extracting features on subset ({len(Xs)} samples)...")
    Fs = extract_features(Xs,
        cell=cfg['hog_cell'], n_bins_hog=cfg['hog_bins'],
        block=cfg['hog_block'], n_bins_col=cfg['col_bins'],
        lbp_radius=cfg['lbp_radius'], lbp_pts=cfg['lbp_pts'])

    # normalise features (important for RBF)
    mu = Fs.mean(axis=0)
    std = Fs.std(axis=0) + 1e-8
    Fs = ((Fs - mu) / std).astype(np.float64)

    col_end = HOG_DIM + COL_DIM
    Fs_hog = Fs[:, :HOG_DIM]
    Fs_col = Fs[:, HOG_DIM:col_end]
    Fs_lbp = Fs[:, col_end:]

    best_acc, best_params = 0.0, {}
    results = []

    for gm_hog in SEARCH_GRID['gamma_mult_hog']:
        for gm_col in SEARCH_GRID['gamma_mult_col']:
            for gm_lbp in SEARCH_GRID['gamma_mult_lbp']:

                gamma_hog = gm_hog / Fs_hog.shape[1]
                gamma_col = gm_col / Fs_col.shape[1]
                gamma_lbp = gm_lbp / Fs_lbp.shape[1]

                print(f"\nTrying: hog={gm_hog}, col={gm_col}, lbp={gm_lbp}")

                K_hog, _ = build_gram_matrices(Fs_hog, Fs_hog[:1], gamma_hog)
                K_col, _ = build_gram_matrices(Fs_col, Fs_col[:1], gamma_col)
                K_lbp, _ = build_gram_matrices(Fs_lbp, Fs_lbp[:1], gamma_lbp)
                K = K_hog + K_col + K_lbp

                for lam in SEARCH_GRID['lam']:
                    t0 = time.time()
                    acc, std_acc = KRROneVsRest.cross_val_accuracy(
                        K, Ys, lam=lam, n_folds=SEARCH_GRID['n_folds'],
                        seed=cfg['seed'])

                    elapsed = time.time() - t0
                    print(f"  lam={lam:.0e} → acc={acc:.4f} ({elapsed:.1f}s)")

                    results.append({
                        'gamma_mult_hog': gm_hog,
                        'gamma_mult_col': gm_col,
                        'gamma_mult_lbp': gm_lbp,
                        'lam': lam,
                        'acc': acc,
                        'std': std_acc
                    })

                    if acc > best_acc:
                        best_acc = acc
                        best_params = {
                            'gamma_mult_hog': gm_hog,
                            'gamma_mult_col': gm_col,
                            'gamma_mult_lbp': gm_lbp,
                            'lam': lam
                        }

    print(f"\nBest CV accuracy: {best_acc:.4f}")
    print("Best params:", best_params)

    pd.DataFrame(results).to_csv('hyperparam_results.csv', index=False)
    return best_params


def main(search=False):
    np.random.seed(CONFIG['seed'])
    t_total = time.time()
    cfg = CONFIG.copy()

    print("Loading data...")
    Xtr, Xte, Ytr = load_data(cfg['data_dir'])
    print(f"Train: {Xtr.shape}, Test: {Xte.shape}")

    if search:
        best = run_search(Xtr, Ytr, cfg)
        cfg.update(best)

    print("Extracting features...")
    Ftr = extract_features(Xtr,
        cell=cfg['hog_cell'], n_bins_hog=cfg['hog_bins'],
        block=cfg['hog_block'], n_bins_col=cfg['col_bins'],
        lbp_radius=cfg['lbp_radius'], lbp_pts=cfg['lbp_pts'])

    Fte = extract_features(Xte,
        cell=cfg['hog_cell'], n_bins_hog=cfg['hog_bins'],
        block=cfg['hog_block'], n_bins_col=cfg['col_bins'],
        lbp_radius=cfg['lbp_radius'], lbp_pts=cfg['lbp_pts'])

    print("Standardising features...")
    Ftr, Fte, *_ = standardise(Ftr, Fte)

    print("Building additive kernel...")
    K_train, K_test = build_additive_kernel(
        Ftr, Fte,
        cfg['gamma_mult_hog'],
        cfg['gamma_mult_col'],
        cfg['gamma_mult_lbp']
    )

    print("Training KRR...")
    clf = KRROneVsRest(lam=cfg['lam'])
    clf.fit(K_train, Ytr)

    train_acc = (clf.predict(K_train) == Ytr).mean()
    print(f"Train acc: {train_acc:.4f}")

    print("Predicting...")
    Yte = clf.predict(K_test)

    out = os.path.join(cfg['data_dir'], cfg['output_file'])
    pd.DataFrame({'Prediction': Yte}).to_csv(out, index_label='Id')

    print(f"Saved to {out}")
    print(f"Total time: {time.time()-t_total:.1f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--search', action='store_true')
    args = parser.parse_args()
    main(search=args.search)