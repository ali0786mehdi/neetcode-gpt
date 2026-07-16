import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(
        self,
        X: NDArray[np.float64],
        weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Forward pass: X · W
        return np.round(np.dot(X, weights), 5)

    def get_error(
        self,
        model_prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64]
    ) -> float:
        # Mean Squared Error
        mse = np.mean((model_prediction - ground_truth) ** 2)
        return round(float(mse), 5)