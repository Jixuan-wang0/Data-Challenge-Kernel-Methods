# Image Classification with KRR + RBF Kernel

## File structure

```
start.py        # main script (run this)
src/
  features.py   # feature extraction (HOG, color, LBP, etc.)
  kernel.py     # RBF kernel computation
  krr.py        # KRR one-vs-rest classifier + cross-validation
```

---

## Overview

The pipeline is:

```
images -> feature extraction -> kernel -> KRR classifier
```

---

## Features (`src/features.py`)

The model uses a combination of different feature types:

| Feature | Description |
|--------|-------------|
| **HOG** | Captures edge and shape information (multi-scale + opponent) |
| **Color features** | HSV spatial pyramid histograms |
| **LBP** | Texture descriptor based on local binary patterns |
| **Patch statistics** | Mean and standard deviation over local patches |

All features are concatenated into a single vector and then standardised.

---

## Kernel (`src/kernel.py`)

I use a standard RBF kernel:

```
K(x, y) = exp(-gamma * ||x - y||^2)
```

Pairwise distances are computed using:

```
||a - b||^2 = ||a||^2 + ||b||^2 - 2a·b
```

The resulting Gram matrices are also normalised.

---

## Model (`src/krr.py`)

I use **Kernel Ridge Regression (KRR)** with a one-vs-rest strategy.

Main idea:
- For each class, learn a function in kernel space
- Stack all classes into one system and solve together

Prediction:
- compute similarity with training samples
- compute scores for each class
- take the class with the highest score

---

## Hyperparameters

| Name | Meaning |
|------|--------|
| `gamma` | kernel bandwidth |
| `lambda` | regularisation strength |

---

## Dependencies

- numpy
- pandas