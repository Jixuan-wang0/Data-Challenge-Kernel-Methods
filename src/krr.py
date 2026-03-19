"""
krr.py - Kernel Ridge Regression, One-vs-Rest multiclass classifier.
"""

import numpy as np


class KRROneVsRest:
    """
    Kernel Ridge Regression with One-vs-Rest multiclass strategy.
    """

    def __init__(self, lam: float = 1.0):
        self.lam    = lam
        self.alphas = None     # (n_classes, N) - one alpha vector per class
        self.classes = None    # sorted array of class labels

    # Training

    def fit(self, K_train: np.ndarray, y: np.ndarray) -> "KRROneVsRest":
        """
        Fit the model.
        """
        N = K_train.shape[0]
        self.classes = np.unique(y)
        n_classes    = len(self.classes)

        # add ridge regularization (scaled by N)
        A = K_train + self.lam * N * np.eye(N, dtype=np.float64)

        # Stack all binary targets into one (N, n_classes) matrix and solve once
        Y = np.where(y[:, None] == self.classes[None, :], 1.0, -1.0)  # (N, C)

        # solve linear system once for all classes
        # np.linalg.solve is O(N^3) but avoids forming the inverse explicitly.
        Alpha = np.linalg.solve(A, Y)   # (N, C)
        self.alphas = Alpha.T           # (C, N) - easier for prediction

        return self

    # Prediction

    def decision_function(self, K_test: np.ndarray) -> np.ndarray:
        """
        Raw real-valued scores for each class.
        """
        # scores[i, c] = K_test[i, :] @ alphas[c, :]
        return K_test @ self.alphas.T   # (M, C)

    def predict(self, K_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        """
        scores = self.decision_function(K_test)  # (M, C)
        return self.classes[np.argmax(scores, axis=1)]

    # Cross-validation

    @staticmethod
    def cross_val_accuracy(K_train: np.ndarray, y: np.ndarray,
                           lam: float, n_folds: int = 5,
                           seed: int = 0) -> tuple[float, float]:
        """
        Stratified k-fold CV accuracy for a given lambda.
        """
        rng     = np.random.RandomState(seed)
        classes = np.unique(y)
        N       = len(y)

        # build simple stratified splits
        fold_ids = np.empty(N, dtype=int)
        for c in classes:
            idx = np.where(y == c)[0]
            idx = rng.permutation(idx)
            for f, i in enumerate(idx):
                fold_ids[i] = f % n_folds

        accs = []
        for f in range(n_folds):
            val_mask = fold_ids == f
            tr_mask  = ~val_mask

            tr_idx  = np.where(tr_mask)[0]
            val_idx = np.where(val_mask)[0]

            K_tr  = K_train[np.ix_(tr_idx,  tr_idx)]
            K_val = K_train[np.ix_(val_idx, tr_idx)]
            y_tr  = y[tr_idx]
            y_val = y[val_idx]

            clf = KRROneVsRest(lam=lam)
            clf.fit(K_tr, y_tr)
            preds = clf.predict(K_val)
            accs.append(float((preds == y_val).mean()))

        return float(np.mean(accs)), float(np.std(accs))
