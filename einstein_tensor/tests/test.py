import unittest
import numpy as np
import einstein_tensor as et

# Next test that we raise a value error when we try to multiply tensors in different frames
# And that we raise a type error when we multiply a tensor with a frame by a tensor without a frame
# There is also a potential issue with the metric tensor: will it always need a frame to work
# Maybe make metric tensor a function that returns a metric tensor with the correct indices and frame

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
        self.assertEqual((self.a * b), vector_squared_value)

    def test_right_multiply_vector_by_one(self) -> None:
        self.assertEqual(self.a * 1, self.a)

    def test_left_multiply_vector_by_one(self) -> None:
        self.assertEqual(1 * self.a, self.a)

    def test_right_multiply_vector_by_two(self) -> None:
        self.assertEqual(self.a * 2, self.a + self.a)

    def test_left_multiply_vector_by_two(self) -> None:
        self.assertEqual(2 * self.a, self.a + self.a)

    def test_right_multiply_vector_by_two_point_five(self) -> None:
        product_tensor = et.Tensor(2.5 * np.array(self.a_value), self.a_indices)
        self.assertEqual(self.a * 2.5, product_tensor)

    def test_left_multiply_vector_by_two_point_five(self) -> None:
        product_tensor = et.Tensor(2.5 * np.array(self.a_value), self.a_indices)
        self.assertEqual(2.5 * self.a, product_tensor)

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

    def test_twice_tensor_minus_itself_equals_itself(self) -> None:
        self.assertEqual((self.a + self.a) - self.a, self.a)

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

    def test_replace_indices_with_same_pattern(self) -> None:
        rank_three_tensor = self.a * et.Tensor(self.a_value, ['_nu']) * et.Tensor(1, ['i'])
        other_rank_three_tensor = et.Tensor(rank_three_tensor.value, ['^alpha', '_beta', 'j'])
        other_rank_three_tensor.reindex_as(['^mu', '_nu', 'i'])
        self.assertEqual(rank_three_tensor, other_rank_three_tensor)

    def test_replace_indices_with_different_number_of_indices_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.a.reindex_as(['^nu', '^rho'])

    def test_replace_indices_with_different_pattern_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.a.reindex_as(['_nu'])

    def test_add_tensor_to_integer_raises_value_error(self) -> None:
        with self.assertRaises(TypeError):
            b = self.a + 3

    def test_add_integer_to_tensor_raises_value_error(self) -> None:
        with self.assertRaises(TypeError):
            b = 3 + self.a

    def test_tensors_with_different_frames_are_not_equal(self):
        a = et.Tensor_with_Frame(self.a_value, self.a_indices, 'A')
        b = et.Tensor_with_Frame(self.a_value, self.a_indices, 'B')
        self.assertNotEqual(a, b)

    def test_scalars_with_different_frames_are_equal(self):
        a = et.Tensor_with_Frame(1, [], 'A')
        b = et.Tensor_with_Frame(1, [], 'B')
        self.assertEqual(a, b)

    def test_scalar_tensor_with_frame_equal_to_tensor(self):
        a = et.Tensor_with_Frame(1, [], 'A')
        b = et.Tensor(1, [])
        self.assertEqual(a, b)

    def test_scalar_tensor_with_frame_equal_to_scalar(self):
        a = et.Tensor_with_Frame(1, [], 'A')
        self.assertEqual(a, 1)

    def test_scalars_tensors_with_different_frames_are_not_equal_if_values_are_unequal(self):
        a = et.Tensor_with_Frame(1, [], 'A')
        b = et.Tensor_with_Frame(2, [], 'B')
        self.assertNotEqual(a, b)

    def test_scalar_tensor_and_tensor_with_frame_are_not_equal_if_values_are_unequal(self):
        a = et.Tensor_with_Frame(1, [], 'A')
        b = et.Tensor(2, [])
        self.assertNotEqual(a, b)

    def test_scalar_tensor_and_tensor_with_frame_are_not_equal_if_values_are_unequal(self):
        a = et.Tensor_with_Frame(1, [], 'A')
        b = et.Tensor(2, [])
        self.assertNotEqual(a, b)

    def test_scalar_tensor_and_scalar_are_not_equal_if_values_are_unequal(self):
        a = et.Tensor_with_Frame(1, [], 'A')
        b = 2
        self.assertNotEqual(a, b)

    def test_that_sum_of_tensors_with_frames_has_same_frame(self):
        a = et.Tensor_with_Frame(self.a_value, self.a_indices, 'A')
        self.assertEqual((a + a).frame, a.frame)

    def test_that_sum_of_tensors_with_different_frames_raises_exception(self):
        a = et.Tensor_with_Frame(self.a_value, self.a_indices, 'A')
        b = et.Tensor_with_Frame(self.a_value, self.a_indices, 'B')
        with self.assertRaises(ValueError):
            c = a + b

    def test_that_sum_of_tensor_with_frame_and_a_tensor_without_frames_raises_exception(self):
        a = et.Tensor_with_Frame(self.a_value, self.a_indices, 'A')
        b = et.Tensor(self.a_value, self.a_indices)
        with self.assertRaises(TypeError):
            c = a + b

    def test_str_tensor_with_frame(self) -> None:
        a = et.Tensor_with_Frame(self.a_value, self.a_indices, 'A')
        self.assertEqual(a.__str__(), str(a.tensor.value) + " " + str(a.tensor.indices) + " " + a.frame)

    def test_repr_tensor_with_frame(self) -> None:
        a = et.Tensor_with_Frame(self.a_value, self.a_indices, 'A')
        self.assertEqual(a.__repr__(), a.__str__())

    def test_neg_tensor_with_frame_plus_tensor_with_frame_equals_zero_tensor(self) -> None:
        zero_list = [0 for x in self.a_value]
        a = et.Tensor_with_Frame(self.a_value, self.a_indices, 'A')
        self.assertEqual(a + -a, et.Tensor_with_Frame(zero_list, self.a_indices, 'A'))

    def test_tensor_with_frame_minus_itself_equals_zero_tensor(self) -> None:
        a = et.Tensor_with_Frame(self.a_value, self.a_indices, 'A')
        zero_list = [0 for x in self.a_value]
        zero_frame_tensor = et.Tensor_with_Frame(zero_list, self.a_indices, 'A')
        self.assertEqual(a - a, zero_frame_tensor)

    def test_twice_tensor_with_frame_minus_itself_equals_itself(self) -> None:
        a = et.Tensor_with_Frame(self.a_value, self.a_indices, 'A')
        self.assertEqual((a + a) - a, a)

    def test_right_multiply_vector_with_frame_by_one(self) -> None:
        a = et.Tensor_with_Frame(self.a_value, self.a_indices, 'A')
        self.assertEqual(a * 1, a)

    def test_left_multiply_vector_with_frame_by_one(self) -> None:
        a = et.Tensor_with_Frame(self.a_value, self.a_indices, 'A')
        self.assertEqual(1 * a, a)

    def test_right_multiply_vector_with_frame_by_two(self) -> None:
        a = et.Tensor_with_Frame(self.a_value, self.a_indices, 'A')
        self.assertEqual(a * 2, a + a)

    def test_left_multiply_vector_with_frame_by_two(self) -> None:
        a = et.Tensor_with_Frame(self.a_value, self.a_indices, 'A')
        self.assertEqual(2 * a, a + a)

    def test_right_multiply_vector_with_frame_by_two_point_five(self) -> None:
        a = et.Tensor_with_Frame(self.a_value, self.a_indices, 'A')
        product_tensor_with_frame = et.Tensor_with_Frame(2.5 * np.array(self.a_value), self.a_indices, 'A')
        self.assertEqual(a * 2.5, product_tensor_with_frame)

    def test_left_multiply_vector_with_frame_by_two_point_five(self) -> None:
        a = et.Tensor_with_Frame(self.a_value, self.a_indices, 'A')
        product_tensor_with_frame = et.Tensor_with_Frame(2.5 * np.array(self.a_value), self.a_indices, 'A')
        self.assertEqual(2.5 * a, product_tensor_with_frame)

    def test_product_of_two_tensors_with_frames_has_same_frame(self) -> None:
        a = et.Tensor_with_Frame(self.a_value, self.a_indices, 'A')
        b = et.Tensor_with_Frame(self.a_value, ['^nu'], 'A')
        self.assertEqual((a * b).frame, 'A')

    def test_contractable_lorentz_indices_in_tensor_are_contracted_upper_lower(self) -> None:
        matrix = np.array([[1, 0], [0, 3]])
        tensor = et.Tensor(matrix, ['^mu', '_mu'])
        self.assertEqual(tensor, 4)

    def test_contractable_lorentz_indices_in_tensor_are_contracted_lower_upper(self) -> None:
        matrix = np.array([[1, 0], [0, 3]])
        tensor = et.Tensor(matrix, ['_mu', '^mu'])
        self.assertEqual(tensor, 4)

    def test_contractable_indices_in_tensor_are_contracted(self) -> None:
        matrix = np.array([[1, 0], [0, 3]])
        tensor = et.Tensor(matrix, ['i', 'i'])
        self.assertEqual(tensor, 4)

    def test_contractable_lorentz_indices_in_tensor_are_contracted_upper_lower_1_3(self) -> None:
        matrix = np.array([[[1, 0]], [[0, 3]]])
        tensor = et.Tensor(matrix, ['^mu', '^nu', '_mu'])
        self.assertEqual(tensor, et.Tensor([4], ['^nu']))

    def test_contracting_more_complicated_index_structure(self) -> None:
        complicated_tensor = et.Tensor([[[[[[[[[[[[1]]]]]]]]]]]],
                                       ['^mu', '_nu', '_rho', 'a', 'b', 'c', 'c', 'a', 'e', '^nu', '^rho', '_mu'])
        simple_tensor = et.Tensor([[1]], ['b', 'e'])
        self.assertEqual(complicated_tensor, simple_tensor)


if __name__ == '__main__':
    unittest.main()
