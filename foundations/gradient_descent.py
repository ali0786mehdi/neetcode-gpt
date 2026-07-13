class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: float) -> float:
        x = init

        for _ in range(iterations):
            x = x - learning_rate * (2 * x)

        return round(x, 5)