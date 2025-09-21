import numpy as np

class SVM:
    """
    SVM lineal soft-margin OVR con SGD + weight decay + promedio de pesos (Averaged SGD).
    - C controla el castigo por violar el margen.
    - Usa datos estandarizados.
    Métodos: fit, predict.
    """
    def __init__(self, lr=0.002, n_iter=150, C=1.0, avg=True):
        self.lr = lr
        self.n_iter = n_iter
        self.C = C
        self.avg = avg
        self.W = None   # (k, d)
        self.b = None   # (k,)
        self.classes = None

    def fit(self, X, y):
        n, d = X.shape
        self.classes = np.unique(y)
        k = len(self.classes)

        W = np.zeros((k, d), dtype=float)
        b = np.zeros(k, dtype=float)

        # acumuladores para averaged SGD
        Wa = np.zeros_like(W)
        ba = np.zeros_like(b)
        count = 0

        for _ in range(self.n_iter):
            idx = np.random.permutation(n)
            Xs, ys = X[idx], y[idx]

            for xi, yi in zip(Xs, ys):
                # entrenamos k clasificadores OVR
                for ci, c in enumerate(self.classes):
                    y_bin = 1.0 if yi == c else -1.0
                    margin = y_bin * (W[ci] @ xi + b[ci])

                    # weight decay (L2) SIEMPRE sobre W
                    W[ci] *= (1.0 - self.lr)

                    if margin < 1.0:
                        # gradiente de la parte hinge
                        W[ci] += self.lr * (self.C * y_bin * xi)
                        b[ci] += self.lr * (self.C * y_bin)

                if self.avg:
                    Wa += W
                    ba += b
                    count += 1

        if self.avg and count > 0:
            self.W = Wa / count
            self.b = ba / count
        else:
            self.W = W
            self.b = b
        return self

    def predict(self, X):
        scores = X @ self.W.T + self.b
        idx = np.argmax(scores, axis=1)
        return self.classes[idx]
