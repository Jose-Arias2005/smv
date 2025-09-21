import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from ml_svm import SVM


def test_l2_parameter_scales_weight_decay():
    X = np.array([[1.0]])
    y = np.array([1])

    model = SVM(lr=0.1, n_iter=2, C=1.0, avg=False, l2=0.5)
    model.fit(X, y)

    assert np.isclose(model.W[0, 0], 0.195, atol=1e-6)
