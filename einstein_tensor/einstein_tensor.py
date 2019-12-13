from __future__ import annotations
import numpy as np


def _is_raised(index: str) -> bool:
    # returns true if index is raised, i.e., begins with ^, false otherwise
    return index[0] == '^'

def _is_lowered(index: str) -> bool:
    # returns true if index is lowered, i.e., begins with _, false otherwise
    return index[0] == '_'

def _is_raised_or_lowered(index: str) -> bool:
    # returns true if index is raised or lowered (begins with '^' or '_', false otherwise
    return _is_raised(index) or _is_lowered(index)

def _is_not_raised_or_lowered(index: str) -> bool:
    # returns true if index is not raised or lowered (i.e. does not begin with '^' or '_'), false otherwise
    return not _is_raised_or_lowered(index)

def _complimentary_index(index: str) -> str:
    # return the string describing a lowered index, if index is a raised index string (and vice versa)
    if index[0] == '^':
        return '_' + index[1:]
    elif index[0] == '_':
        return '^' + index[1:]
    return index


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
    """

    def __init__(self, value, indices) -> None:
        self.value = np.array(value)
        self.indices = np.array(indices)
        self._raise_value_error_for_illegal_repeated_indices()
        self._contract_indices()

    def __eq__(self, other) -> bool:
        try:
            if self.is_scalar(self):
                return self.value == other
            else:
                return self._check_tensor_members_are_equal(other)
        except AttributeError:
            return False

    def __str__(self) -> str:
        return str(self.value) + " " + str(self.indices)

    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, other: Tensor) -> Tensor:
        try:
            if self.indices != other.indices:
                raise ValueError("Tensors have different indices so cannot be added.")
            return Tensor(self.value + other.value, self.indices)
        except AttributeError:
            raise TypeError("unsupported operand type(s) for +: 'Tensor' and " + str(type(other)))

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
        tuple_of_indices = self._coax_singleton_to_tuple(tuple_of_indices)
        self._raise_value_error_if_index_tuple_has_wrong_length(tuple_of_indices, value_error_string)
        list_of_indices = self._make_list_of_indices_in_sliced_tensor(tuple_of_indices, value_error_string)
        return Tensor(self.value[tuple_of_indices], list_of_indices)

    def is_scalar(self, tensor) -> bool:
        return tensor.value.shape == ()

    def reindex_as(self, new_indices):
        value_error_string = 'Cannot change number of indices/ dimensionality of tensor through reindexing.'
        self._raise_value_error_if_index_tuple_has_wrong_length(new_indices, value_error_string)
        for old, new in zip(self.indices, new_indices):
            if _is_raised(old) and not _is_raised(new):
                raise ValueError('Cannot change raised index to other kind of index through reindexing.')
            if _is_lowered(old) and not _is_lowered(new):
                raise ValueError('Cannot change lowered index to other kind of index through reindexing.')
            if _is_not_raised_or_lowered(old) and _is_raised_or_lowered(new):
                raise ValueError('Cannot change generic index to raised or lowered index through reindexing.')
        self.indices = new_indices

    # __init__
    def _contract_indices(self):
        found_indices_to_contract = True
        while found_indices_to_contract:
            found_in_this_loop = False
            for i_first_index, first_index in enumerate(self.indices):
                for i_second_index, second_index in enumerate(self.indices[i_first_index + 1:], i_first_index + 1):
                    if first_index == _complimentary_index(second_index):
                        self._contract_specific_indices(i_first_index, i_second_index)
                        found_in_this_loop = True
                        break
                if found_in_this_loop:
                    break
            else:
                found_indices_to_contract = False

    def _contract_specific_indices(self, i_first_index, i_second_index):
        contract_index_numbers = [i_first_index, i_second_index]
        self.value = np.trace(self.value, axis1=i_first_index, axis2=i_second_index)
        self.indices = np.array([index for i_index, index in enumerate(self.indices)
                                 if i_index not in contract_index_numbers])

    def _raise_value_error_for_illegal_repeated_indices(self) -> None:
        if len(self.indices) <= 1:
            return
        index_count_dict = self._count_indices()
        self._check_for_repeated_indices_and_raise_value_error(index_count_dict)

    def _check_for_repeated_indices_and_raise_value_error(self, index_count_dict):
        for index, count in index_count_dict.items():
            if count > 1:
                if _is_raised(index):
                    raise ValueError('Cannot have identical raised indices.')
                elif _is_lowered(index):
                    raise ValueError('Cannot have identical lowered indices.')
                else:
                    if count > 2:
                        raise ValueError('Cannot have three identical indices.')

    def _count_indices(self):
        index_count_dict = {}
        for index in self.indices:
            if index in index_count_dict:
                index_count_dict[index] += 1
            else:
                index_count_dict[index] = 1
        return index_count_dict

    def _check_tensor_members_are_equal(self, other):
        equal_values = np.array_equal(self.value, other.value)
        equal_indices = np.array_equal(self.indices, other.indices)
        return equal_values and equal_indices

    def _coax_singleton_to_tuple(self, tuple_of_indices):
        if type(tuple_of_indices) is not tuple:
            tuple_of_indices = (tuple_of_indices,)
        return tuple_of_indices

    def _raise_value_error_if_index_tuple_has_wrong_length(self, tuple_of_indices, value_error_string):
        if len(tuple_of_indices) != len(self.indices):
            raise ValueError(value_error_string)

    def _make_list_of_indices_in_sliced_tensor(self, tuple_of_indices, value_error_string):
        list_of_indices = []
        for i_index, index in enumerate(tuple_of_indices):
            self._process_slice_index(i_index, index, list_of_indices, value_error_string)
        return list_of_indices

    def _process_slice_index(self, i_index, index, list_of_indices, value_error_string):
        if type(index) is int:
            pass
        elif type(index) is slice:
            self._add_index_to_list_if_valid_slice(i_index, index, list_of_indices, value_error_string)
        else:
            raise ValueError(value_error_string)

    def _add_index_to_list_if_valid_slice(self, i_index, index, list_of_indices, value_error_string):
        if index.start is None and index.stop is None and index.step is None:
            list_of_indices.append(self.indices[i_index])
        else:
            raise ValueError(value_error_string)

    def _multiply_by_tensor(self, other: Tensor) -> Tensor:
        # create a _TensorMultiplier object and use it to perform the multiplication
        value = np.multiply.outer(self.value, other.value)
        indices = list(self.indices) + list(other.indices)
        return Tensor(value, indices)

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


class Tensor_with_Frame(Tensor):
    """
    Give Tensor a frame attribute which helps us to keep track of reference frames.

    Initialized with
    Args:
        value: a numpy array, list, or tuple containing the values in the tensor
        indices: a list or numpy array of strings describing the indices.
        frame: generally a string indicating the frame name

    These arguments are cast into numpy arrays which are the public attributes of the class.
    """

    def __init__(self, value, indices, frame):
        self.tensor = Tensor(value, indices)
        self.frame = frame

    @classmethod
    def from_tensor(cls, tensor, frame):
        return cls(tensor.value, tensor.indices, frame)

    def _cast_scalar_to_my_frame(self, scalar):
        if hasattr(scalar, 'tensor'):
            return Tensor_with_Frame(scalar.tensor.value, self.tensor.indices, self.frame)
        elif hasattr(scalar, 'value'):
            return Tensor_with_Frame(scalar.value, [], self.frame)
        else:
            return Tensor_with_Frame(scalar, [], self.frame)

    def tensor_equals(self, other):
        try:
            return self.tensor == other.tensor and self.frame == other.frame
        except AttributeError:
            return False

    def __eq__(self, other) -> bool:
        if self.is_scalar(self.tensor):
            new_other = self._cast_scalar_to_my_frame(other)
            return self.tensor_equals(new_other)
        else:
            return self.tensor_equals(other)

    def _check_that_other_frame_exists_and_is_same(self, other) -> None:
        try:
            if self.frame != other.frame:
                raise ValueError("Cannot add tensors in different frames.")
        except AttributeError:
            raise TypeError('Cannot add Tensor_with_Frame to Tensor (without frame).')

    def __add__(self, other: Tensor_with_Frame) -> Tensor_with_Frame:
        self._check_that_other_frame_exists_and_is_same(other)
        return Tensor_with_Frame.from_tensor(self.tensor + other.tensor, self.frame)

    def __str__(self) -> str:
        return str(self.tensor.value) + " " + str(self.tensor.indices) + " " + self.frame

    def __neg__(self) -> Tensor_with_Frame:
        return Tensor_with_Frame.from_tensor(-self.tensor, self.frame)

    def __rmul__(self, other) -> Tensor_with_Frame:
        return Tensor_with_Frame.from_tensor(other * self.tensor, self.frame)

    def __mul__(self, other) -> Tensor_with_Frame:
        if hasattr(other, 'frame'):
            if other.frame == self.frame:
                return Tensor_with_Frame.from_tensor(self.tensor * other.tensor, self.frame)
            raise ValueError("Cannot multiply tensors in different frames.")
        else:
            if hasattr(other, 'value') and hasattr(other, 'indices'):
                raise TypeError('Cannot multiply Tensor_with_Frame and Tensor (without frame).')
            else:
                return Tensor_with_Frame.from_tensor(self.tensor * other, self.frame)

    def __getitem__(self, tuple_of_indices):
        return Tensor_with_Frame.from_tensor(super().__getitem__(tuple_of_indices), self.frame)

    def reindex_as(self, new_indices):
        return Tensor_with_Frame.from_tensor(super().reindex_as(new_indices), self.frame)

    def reframe_as(self, new_frame):
        return Tensor_with_Frame.from_tensor(self.tensor, new_frame)