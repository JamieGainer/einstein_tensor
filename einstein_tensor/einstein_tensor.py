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

    def __mul__(self, other):
        try:
            if self.there_are_repeated_indices(other):
                raise ValueError("Cannot multiply tensors with identical indices.")
            return self.multiply_tensors(other)
        except AttributeError:
            return self.left_multiply_scalar(other)

    def __rmul__(self, other):
        return self.right_multiply_scalar(other)

    def __str__(self):
        return str(self.value) + " " + str(self.indices)

    def __repr__(self):
        return self.__str__()

    def multiply_tensors(self, other):
        self.initialize_data_structures_for_multiplication(other)
        self.fill_in_data_structures_for_multiplication()
        new_value, new_indices = self.obtain_new_value_and_indices(other)
        self.remove_data_structures()
        return Tensor(new_value, new_indices)

    def left_multiply_scalar(self, other):
        new_value = self.value * other
        new_indices = self.indices
        return Tensor(new_value, new_indices)

    def right_multiply_scalar(self, other):
        new_value = other * self.value
        new_indices = self.indices
        return Tensor(new_value, new_indices)

    def obtain_new_value_and_indices(self, other):
        new_value = np.tensordot(self.value, other.value, axes=(self.my_contract_indices, self.other_contract_indices))
        new_indices = [index for index in self.indices if index not in self.my_indices_to_remove]
        new_indices += [index for index in other.indices if index not in self.other_indices_to_remove]
        return new_value, new_indices

    def complimentary_index(self, index: str) -> str:
        compliment_dict = {'^': '_', '_': '^'}
        for index_start in compliment_dict:
            if index.startswith(index_start):
                return compliment_dict[index_start] + index[len(index_start):]
        raise ValueError("Index must be either raised (with initial '^') or lowered " +
                             "(with initial ('_').")

    def there_are_repeated_indices(self, other) -> bool:
        my_indices = set(self.indices)
        other_indices = set(other.indices)
        return len(my_indices.union(other_indices)) < len(my_indices) + len(other.indices)

    def initialize_data_structures_for_multiplication(self, other):
        self.my_contract_indices, self.other_contract_indices = [], []
        self.my_indices_to_remove, self.other_indices_to_remove = set([]), set([])
        self.other_indices_dict = {index: i_index for i_index, index in enumerate(other.indices)}

    def fill_in_data_structures_for_multiplication(self):
        for i_index, index in enumerate(self.indices):
            search_index = self.complimentary_index(index)
            if search_index in self.other_indices_dict:
                self.update_contracted_index_lists(i_index, index, search_index)

    def update_contracted_index_lists(self, i_index: int, index: str, search_index: str):
        self.my_contract_indices.append(i_index)
        self.other_contract_indices.append(self.other_indices_dict[search_index])
        self.my_indices_to_remove.add(index)
        self.other_indices_to_remove.add(search_index)

    def remove_data_structures(self):
        del self.my_contract_indices
        del self.other_contract_indices
        del self.my_indices_to_remove
        del self.other_indices_to_remove
        del self.other_indices_dict
