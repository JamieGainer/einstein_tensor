import unittest
import numpy as np
import einstein_tensor as et

class TestEinsteinTensor(unittest.TestCase):

    a_value = [1, 0, 0, 0]
    assert type(a_value) in [type([]), type((0,))]
    assert any([x != 0 for x in a_value])
    a_indices = ['^mu']
    assert len(a_indices) == 1
    assert a_indices[0][0] == '^'
    a = et.Tensor(a_value, a_indices)

    def test_identical_tensors_are_equal(self) -> None:
        b = et.Tensor(self.a_value, self.a_indices)
        self.assertEqual(self.a, b)

    def test_tensors_with_different_shapes_are_not_equal(self) -> None:
        b = et.Tensor(self.a_value * 2, self.a_indices)
        self.assertNotEqual(self.a, b)

    def test_tensors_with_different_values_are_not_equal(self) -> None:
        b = et.Tensor(2 * np.array(self.a_value), self.a_indices)
        self.assertNotEqual(self.a, b)

    def test_tensors_with_different_indices_are_not_equal(self) -> None:
        b = et.Tensor(self.a_value, [self.a_indices[0] + '*'])
        self.assertNotEqual(self.a, b)

    def test_adding_two_tensors(self) -> None:
        b = et.Tensor(2 * np.array(self.a_value), self.a_indices)
        self.assertEqual(self.a + self.a, b)

    def test_exception_raised_when_adding_tensors_with_different_indices(self) -> None:
        b = et.Tensor(self.a_value, [self.a_indices[0] + '*'])
        with self.assertRaises(ValueError):
            c = self.a + b

    def test_cannot_multiply_tensors_with_the_same_index(self) -> None:
        b = et.Tensor(self.a_value, self.a_indices)
        with self.assertRaises(ValueError):
            c = self.a * b

    def test_multiply_vector_and_1_form(self) -> None:
        lowered_indices = [self.a_indices[0].replace('^', '_')]
        b = et.Tensor(self.a_value, lowered_indices)
        vector_squared_value = self.a_value[0]**2 - sum([x**2 for x in self.a_value[1:]])
        self.assertEqual((self.a * b).value, vector_squared_value)

    def test_right_multiply_vector_by_constant(self) -> None:
        self.assertEqual(self.a * 1, self.a)

    def test_left_multiply_vector_by_constant(self) -> None:
        self.assertEqual(1 * self.a, self.a)

    def test_str_tensor(self) -> None:
        self.assertEqual(self.a.__str__(), str(self.a.value) + " " + str(self.a.indices))

    def test_repr_tensor(self) -> None:
        self.assertEqual(self.a.__repr__(), self.a.__str__())

    def test_neg_tensor_plus_tensor_equals_zero_tensor(self) -> None:
        zero_list = [0 for x in self.a_value]
        self.assertEqual(self.a + -self.a, et.Tensor(zero_list, self.a_indices))

    def test_tensor_minus_itself_equals_zero_tensor(self) -> None:
        zero_list = [0 for x in self.a_value]
        self.assertEqual(self.a - self.a, et.Tensor(zero_list, self.a_indices))

    def test_list_passed_to_tensor_as_only_input_raises_exception(self) -> None:
        with self.assertRaises(ValueError):
            b = self.a[[0]]

    def test_list_passed_to_tensor_as_one_input_raises_exception(self) -> None:
        rank_two_tensor = self.a * et.Tensor(self.a_value, ['^nu'])
        with self.assertRaises(ValueError):
            b = rank_two_tensor[[0],:]

    def test_wrong_number_of_indices_passed_to_tensor_as_slice_raises_exception(self) -> None:
        rank_two_tensor = self.a * et.Tensor(self.a_value, ['^nu'])
        with self.assertRaises(ValueError):
            b = rank_two_tensor[:]

    def test_wrong_number_of_indices_passed_to_tensor_as_index_raises_exception(self) -> None:
        rank_two_tensor = self.a * et.Tensor(self.a_value, ['^nu'])
        with self.assertRaises(ValueError):
            b = rank_two_tensor[0]

    def test_rank_zero_tensor_equals_corresponding_scalar(self) -> None:
        three = et.Tensor(3, [])
        self.assertEqual(three, 3)

    def test_one_slice_then_one_slice_in_rank_two_tensor(self) -> None:
        rank_two_tensor = self.a * et.Tensor(self.a_value, ['^nu'])
        self.assertEqual(rank_two_tensor[:, 0], self.a)

    def test_one_index_then_one_slice_in_rank_two_tensor(self) -> None:
        rank_two_tensor = self.a * et.Tensor(self.a_value, ['^nu'])
        a_nu = et.Tensor(self.a_value, ['^nu'])
        self.assertEqual(rank_two_tensor[0, :], a_nu)

    def test_two_indices_in_rank_two_tensor(self) -> None:
        rank_two_tensor = self.a * et.Tensor(self.a_value, ['^nu'])
        self.assertEqual(rank_two_tensor[0, 0], 1)

    def test_two_slices_in_rank_two_tensor(self) -> None:
        rank_two_tensor = self.a * et.Tensor(self.a_value, ['^nu'])
        self.assertEqual(rank_two_tensor[:, :], rank_two_tensor)


if __name__ == '__main__':
    unittest.main()
