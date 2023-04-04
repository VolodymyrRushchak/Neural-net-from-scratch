import numpy as np


def sigma(x: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-x))


def sigma_derivative(x: np.ndarray) -> np.ndarray:
    return x * (1 - x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1 - x * x


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return x > 0
