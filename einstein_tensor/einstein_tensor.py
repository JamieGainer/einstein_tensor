import numpy as np

class Tensor():

    def __init__(self, value, indices):
        self.value = np.array(value)
        self.indices = np.array(indices)

    def __eq__(self, other):
        return np.all(self.value == other.value) and np.all(self.indices == other.indices)

    def __add__(self, other):
        if self.indices != other.indices:
            raise ValueError("Tensors have different indices so cannot be added.")
        return Tensor(self.value + other.value, self.indices)