from __future__ import annotations
import numpy as np


class Tensor():
    """
    Creates a tensor with indices.  Multiplication of these tensors automatically follows the Einstein summation
    convention including contracting a raised index in the first tensor with a lowered index in the second
    tensor.

    Initialized with
    Args:
        value: a numpy array, list, or tuple containing the values in the tensor
        indices: a list or numpy array of strings describing the indices.

    These arguments are cast into numpy arrays which are the public attributes of the class.
    ddd"""

    def __init__(self, value, indices) -> None:
        self.value = np.array(value)
        self.indices = np.array(indices)

    def __eq__(self, other) -> bool:
        try:
            equal_values = np.array_equal(self.value, other.value)
            equal_indices = np.array_equal(self.indices, other.indices)
            return equal_values and equal_indices
        except:
            if self.value.shape == ():
                return self.value == other
            return False

    def __str__(self) -> str:
        return str(self.value) + " " + str(self.indices)

    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, other: Tensor) -> Tensor:
        if self.indices != other.indices:
            raise ValueError("Tensors have different indices so cannot be added.")
        return Tensor(self.value + other.value, self.indices)

    def __sub__(self, other: Tensor) -> Tensor:
        return self + -other

    def __neg__(self) -> Tensor:
        return Tensor(-self.value, self.indices)

    def __rmul__(self, other) -> Tensor:
        return self._left_multiply_tensor_by_scalar(other)

    def __mul__(self, other) -> Tensor:
        try:
            return self._multiply_by_tensor(other)
        except AttributeError:
            return self._right_multiply_tensor_by_scalar(other)

    def __getitem__(self, tuple_of_indices):
        value_error_string = "Indices for tensor must be an integer or an empty slice for each index."
        if type(tuple_of_indices) is not tuple:
            tuple_of_indices = (tuple_of_indices, )
        if len(tuple_of_indices) != len(self.indices):
            raise ValueError(value_error_string)
        list_of_indices = []
        for i_index, index in enumerate(tuple_of_indices):
            if type(index) is int:
                pass
            elif type(index) is slice:
                if index.start is None and index.stop is None and index.step is None:
                    list_of_indices.append(self.indices[i_index])
                else:
                    raise ValueError(value_error_string)
            else:
                raise ValueError(value_error_string)
        return Tensor(self.value[tuple_of_indices], list_of_indices)


    def _multiply_by_tensor(self, other: Tensor) -> Tensor:
        # create a _TensorMultiplier object and use it to perform the multiplication
        tensor_multiplier = _TensorMultiplier(self, other)
        return tensor_multiplier._result()

    def _right_multiply_tensor_by_scalar(self, other) -> Tensor:
        # perform tensor * scalar multiplication
        new_value = self.value * other
        new_indices = self.indices
        return Tensor(new_value, new_indices)

    def _left_multiply_tensor_by_scalar(self, other) -> Tensor:
        # perform scalar * tensor multiplication
        new_value = other * self.value
        new_indices = self.indices
        return Tensor(new_value, new_indices)


class _TensorMultiplier():
    # Internal class to perform Einstein summation convention tensor multiplications with (possibly)
    # raised or lowered indices

    def __init__(self, tensor_a: Tensor, tensor_b: Tensor) -> None:
        # class takes two instantiations of Tensor as its arguments
        self._a, self._b = tensor_a, tensor_b
        self._a_contract_indices, self._b_contract_indices, self._a_keep_indices= [], [], []
        self._b_remove_indices = set([])
        self._b_indices_index = {index: i_index for i_index, index in enumerate(tensor_b.indices)}

    def _result(self) -> Tensor:
        # returns the Tensor found by multiplying self._a and self._b
        self._process_indices()
        new_value = np.tensordot(self._a.value, self._b.value,
                                 axes=(self._a_contract_indices, self._b_contract_indices))
        new_indices = self._a_keep_indices + self._b_keep_indices
        return Tensor(new_value, new_indices)

    def _process_indices(self) -> None:
        # prepares for the multiplication by making lists describing the indices to keep and
        # those to sum over
        for i_index, index in enumerate(self._a.indices):
            self._check_for_repeated_raised_or_lowered_index(index)
            complimentary_index = self._obtain_complimentary_method(index)
            self._update_index_structures(i_index, index, complimentary_index)
        self._b_keep_indices = [index for index in self._b.indices if index not in self._b_remove_indices]

    def _check_for_repeated_raised_or_lowered_index(self, index: str) -> None:
        # return a ValueError if a raised or lowered index occurs in both self._a and self._b
        if not self._is_raised_or_lowered(index):
            return
        if index in self._b_indices_index:
            raise ValueError("Cannot multiply tensors with identical raised or lowered indices.")

    def _is_raised_or_lowered(self, index: str) -> bool:
        # returns true if coordinate corresponds to a raised ('^') or lowered ('_') index, false otherwise
        return index[0] in ['^', '_']

    def _obtain_complimentary_method(self, index: str) -> str:
        # return the string describing a lowered index, if index is a raised index string (and vice versa)
        # if the input string in "index" is neither raised or lowered, return index
        complimentary_index = index
        if self._is_raised_or_lowered(index):
            complimentary_index = self._complimentary_index(index)
        return complimentary_index

    def _complimentary_index(self, index: str) -> str:
        # return the string describing a lowered index, if index is a raised index string (and vice versa)
        if index[0] == '^':
            return '_' + index[1:]
        elif index[0] == '_':
            return '^' + index[1:]
        return index

    def _update_index_structures(self, i_index: int, index: str, complimentary_index) -> None:
        # fills in the appropriate index structures depending on whether or not
        # complimentary_index is one of self._b's indices
        if complimentary_index in self._b_indices_index:
            self._index_in_a_compliment_in_b(i_index, complimentary_index)
        else:
            self._index_not_in_b(index)

    def _index_in_a_compliment_in_b(self, i_index: int, complimentary_index: str) -> None:
        # fills in the appropriate index structures when complimentary_index is one
        # of self._b's indices
        self._a_contract_indices.append(i_index)
        self._b_contract_indices.append(self._b_indices_index[complimentary_index])
        self._b_remove_indices.add(complimentary_index)

    def _index_not_in_b(self, index: str) -> None:
        # fills int the appropriate index structure when complimentary_index is not
        # one of self._b's indices
        self._a_keep_indices.append(index)

