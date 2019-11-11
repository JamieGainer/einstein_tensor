import unittest
import numpy as np
import einstein_tensor as et

class TestEinsteinTensor(unittest.TestCase):

    a_value = [1, 0, 0, 0]
    assert type(a_value) in [type([]), type((0,))]
    assert any([x != 0 for x in a_value])
    a_indices = ['^mu']
    a = et.Tensor(a_value, a_indices)

    def test_identical_tensors_are_equal(self):
        b = et.Tensor(self.a_value, self.a_indices)
        self.assertEqual(self.a, b)

    def test_tensors_with_different_shapes_are_not_equal(self):
        b = et.Tensor(self.a_value * 2, self.a_indices)
        self.assertNotEqual(self.a, b)

    def test_tensors_with_different_values_are_not_equal(self):
        b = et.Tensor(2 * np.array(self.a_value), self.a_indices)
        self.assertNotEqual(self.a, b)

    def test_tensors_with_different_indices_are_not_equal(self):
        b = et.Tensor(self.a_value, [self.a_indices[0] + '*'])
        self.assertNotEqual(self.a, b)

    def test_adding_two_tensors(self):
        b = et.Tensor(2 * np.array(self.a_value), self.a_indices)
        self.assertEqual(self.a + self.a, b)

    def test_exception_raised_when_adding_tensors_with_different_indices(self):
        b = et.Tensor(self.a_value, [self.a_indices[0] + '*'])
        with self.assertRaises(ValueError):
            c = self.a + b

if __name__ == '__main__':
    unittest.main()
