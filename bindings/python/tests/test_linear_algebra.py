import unittest

from contexts import contexts
import vinnpy
from nose_parameterized import parameterized


class test_linear_algebra(unittest.TestCase):
    def assertMatrixEquals(self, expected, actual):
        ''' Assert that two matrices have the same dimensions and the values '''
        self.assertEqual(expected.row_count(), actual.row_count())
        self.assertEqual(expected.column_count(), actual.column_count())

        for row in range(expected.row_count()):
            for column in range(expected.column_count()):
                self.assertEqual(expected[row][column], actual[row][column])

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_create_matrix(self, context):
        '''
        Test all c++ matrix constructors are accessible from Python
        '''
        expected = vinnpy.matrix(context, 2, 3, 3.0)
        self.assertEqual(2, expected.row_count())
        self.assertEqual(3, expected.column_count())
        self.assertEqual(3.0, expected[0][0])
        self.assertEqual(3.0, expected[0][1])
        self.assertEqual(3.0, expected[0][2])
        self.assertEqual(3.0, expected[1][0])
        self.assertEqual(3.0, expected[1][1])
        self.assertEqual(3.0, expected[1][2])

        # construct using a 2 dimensional array
        self.assertMatrixEquals(expected, vinnpy.matrix(context, [
            [3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0]
        ]))

        # construct using default value = 0.0
        self.assertMatrixEquals(vinnpy.matrix(context, 2, 3, 0.0), vinnpy.matrix(context, 2, 3))
        # construct passing size as a tuple
        self.assertMatrixEquals(vinnpy.matrix(context, (2, 3), 24.0), vinnpy.matrix(context, 2, 3, 24.0))
        # construct passing size as a tuple and default value = 0.0
        self.assertMatrixEquals(vinnpy.matrix(context, (2, 3), 0.0), vinnpy.matrix(context, 2, 3))

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_matrix_multiply(self, context):
        a = vinnpy.matrix(context, 5, 7, 2.0)
        b = vinnpy.matrix(context, 7, 5, 3.0)
        c = a * b
        expected = vinnpy.matrix(context, 5, 5, 42.0)
        self.assertMatrixEquals(expected, c)

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_throw_incompatible_dimensions_can_be_caught(self, context):
        '''
        Test that c++ vi::la::incompatible_dimensions exception is translated
        into python exception that can be caught
        '''
        a = vinnpy.matrix(context, 5, 7, 2.0)
        b = vinnpy.matrix(context, 5, 5, 3.0)
        self.assertRaises(vinnpy.incompatible_dimensions, lambda: a * b)

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_throw_out_of_range_can_be_caught(self, context):
        '''
        Test that c++ std::out_of_range exception is translated
        into python IndexError that can be caught
        '''
        a = vinnpy.matrix(context, 5, 5)
        self.assertRaises(IndexError, lambda: a.row(10))

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_matrix_concatenate(self, context):
        left = vinnpy.matrix(context, 3, 2, 2.0)
        right = vinnpy.matrix(context, 3, 3, 2.0)
        expected = vinnpy.matrix(context, 3, 5, 2.0)
        combined = left.concatenate(right)
        self.assertMatrixEquals(expected, combined)

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_matrix_mutation(self, context):
        a = vinnpy.matrix(context, 3, 3, 2.0)
        a[0][0] = 7.0
        self.assertEqual(7.0, a[0][0])

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_matrix_addition(self, context):
        a = vinnpy.matrix(context, 2, 2, 2.0)
        expected = vinnpy.matrix(context, [[4.0, 4.0], [4.0, 4.0]])
        b = a + a
        self.assertMatrixEquals(expected, b)

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_matrix_subtraction(self, context):
        a = vinnpy.matrix(context, 2, 2, 2.0)
        b = vinnpy.matrix(context, 2, 2, 3.0)
        c = a - b
        expected = vinnpy.matrix(context, [[-1.0, -1.0], [-1.0, -1.0]])
        self.assertMatrixEquals(expected, c)

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_matrix_scalar_multiply(self, context):
        a = vinnpy.matrix(context, 2, 2, 2.0)
        b = a * 42.0
        expected = vinnpy.matrix(context, 2, 2, 84.0)
        self.assertMatrixEquals(expected, b)

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_matrix_elementwise_product(self, context):
        a = vinnpy.matrix(context, 2, 2, 2.0)
        b = a.elementwise_product(a)
        expected = vinnpy.matrix(context, 2, 2, 4.0)
        self.assertMatrixEquals(expected, b)

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_matrix_scalar_division(self, context):
        a = vinnpy.matrix(context, 2, 2, 2.0)
        b = a / 2.0
        expected = vinnpy.matrix(context, 2, 2, 1.0)
        self.assertMatrixEquals(expected, b)

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_matrix_single_row_access(self, context):
        a = vinnpy.matrix(context, 2, 2, 2.0)
        b = a.row(0)
        expected = vinnpy.matrix(context, 1, 2, 2.0)
        self.assertMatrixEquals(expected, b)

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_matrix_multi_row_access(self, context):
        a = vinnpy.matrix(context, 5, 5, 2.0)
        b = a.rows(1, 3)
        expected = vinnpy.matrix(context, 3, 5, 2.0)
        self.assertMatrixEquals(expected, b)

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_matrix_single_column_access(self, context):
        a = vinnpy.matrix(context, 5, 5, 2.0)
        b = a.column(1)
        expected = vinnpy.matrix(context, 5, 1, 2.0)
        self.assertMatrixEquals(expected, b)

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_matrix_multi_column_access(self, context):
        a = vinnpy.matrix(context, 5, 5, 2.0)
        b = a.columns(2, 4)
        expected = vinnpy.matrix(context, 5, 3, 2.0)
        self.assertMatrixEquals(expected, b)
