import unittest

import vinnpy
import numpy
from contexts import contexts
from nose_parameterized import parameterized


class test_numpy_integration(unittest.TestCase):
    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_create_with_numpy_matrix_succeeds(self, context):
        a = vinnpy.matrix(context, numpy.matrix([[1.0, 2.0], [3.0, 4.0]], dtype='float32'))
        self.assertEqual(a[0][0], 1.0)
        self.assertEqual(a[0][1], 2.0)
        self.assertEqual(a[1][0], 3.0)
        self.assertEqual(a[1][1], 4.0)

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_modifying_created_matrix_does_not_mutate_original(self, context):
        '''
        Ensure matrix constructor does not reference the original numpy array's
        memory and is mapped using the correct row-major memory layout
        '''
        array = numpy.matrix([[1.0, 2.0], [3.0, 4.0]], dtype='float32')
        a = vinnpy.matrix(context, array)
        a[0][0] = 42.0
        self.assertEqual(array[0, 0], 1.0)
        self.assertEqual(array[0, 1], 2.0)
        self.assertEqual(array[1, 0], 3.0)
        self.assertEqual(array[1, 1], 4.0)

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_can_access_rows_as_numpy_arrays(self, context):
        '''
        Ensure rows are mapped to numpy arrays when accessed
        '''
        a = vinnpy.matrix(context, 2, 4, 7.0)

        self.assertEqual(4, len(a[0]))
        self.assertIsInstance(a[0], numpy.ndarray)
        numpy.testing.assert_array_equal(numpy.array([7.0, 7.0, 7.0, 7.0]), a[0])

        self.assertEqual(4, len(a[1]))
        self.assertIsInstance(a[1], numpy.ndarray)
        numpy.testing.assert_array_equal(numpy.array([7.0, 7.0, 7.0, 7.0]), a[1])

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_mutating_matrix_through_rows_accessed_as_numpy_arrays(self, context):
        '''
        Ensure the actual elements of the matrix can be mutated using
        a numpy array instead of just temporary copies
        '''
        a = vinnpy.matrix(context, 2, 4, 7.0)
        row = a[0]
        row[0] = 42.0
        self.assertEqual(42.0, a[0][0])

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_len_returns_row_count(self, context):
        a = vinnpy.matrix(context, 4, 2, 7.0)
        self.assertEqual(4, len(a))

    @parameterized.expand(contexts().enumerate, contexts().name)
    def test_row_iteration(self, context):
        row_count = 4
        a = vinnpy.matrix(context, row_count, 2, 7.0)
        rows_iterated = 0
        for row in a:
            rows_iterated += 1
        self.assertEqual(row_count, rows_iterated)
