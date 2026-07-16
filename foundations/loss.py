import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1 - eps)

        loss = -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

        return round(float(loss), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1 - eps)

        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

        return round(float(loss), 4)