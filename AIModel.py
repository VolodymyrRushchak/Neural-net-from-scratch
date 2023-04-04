import copy
from typing import List

import numpy as np

from mymath import sigma


class Model:
    def __init__(self, sizes: List[int], activation_function):
        self.weights = np.array([np.random.normal(size=(a, b)) / np.sqrt(a) for a, b in zip(sizes[:-1], sizes[1:])], dtype=object)
        self.biases = np.array([np.zeros(size) for size in sizes[1:]], dtype=object)
        self.layers = np.array([np.zeros(size) for size in sizes], dtype=object)
        self.activation_function = activation_function

    def evaluate(self, _input: np.ndarray) -> np.ndarray:
        self.layers[0] = copy.deepcopy(_input)
        count = 1
        for weight_matrix, bias_matrix in list(zip(self.weights, self.biases))[:-1]:
            _input = self.activation_function(_input @ weight_matrix + bias_matrix)
            self.layers[count] = copy.deepcopy(_input)
            count += 1
        _input = sigma(_input @ self.weights[-1] + self.biases[-1])
        self.layers[count] = copy.deepcopy(_input)
        return _input
