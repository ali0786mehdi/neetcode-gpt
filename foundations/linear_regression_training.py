import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(
        self,
        model_prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64],
        N: int,
        X: NDArray[np.float64],
        desired_weight: int
    ) -> float:
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(
        self,
        X: NDArray[np.float64],
        weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        weights = initial_weights.astype(np.float64).copy()
        N = len(X)

        for _ in range(num_iterations):
            predictions = self.get_model_prediction(X, weights)

            for j in range(len(weights)):
                gradient = self.get_derivative(
                    predictions,
                    Y,
                    N,
                    X,
                    j
                )
                weights[j] -= self.learning_rate * gradient

        return np.round(weights, 5)