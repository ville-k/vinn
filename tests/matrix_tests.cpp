#include "test.h"
#include "vi/la/matrix.h"

using namespace std;
using namespace vi::la;

class matrix_tests : public ::testing::TestWithParam<vi::la::context*> {};
INSTANTIATE_TEST_CASE_P(context, matrix_tests, ::testing::ValuesIn(test::all_contexts()));

TEST_P(matrix_tests, invalid_construction) {
  EXPECT_THROW(matrix(*GetParam(), 0U, 0U), incompatible_dimensions);
  EXPECT_THROW(matrix(*GetParam(), 1U, 0U), incompatible_dimensions);
  EXPECT_THROW(matrix(*GetParam(), 0U, 1U), incompatible_dimensions);
  EXPECT_THROW(matrix(*GetParam(), {}), incompatible_dimensions);
  EXPECT_THROW(matrix(*GetParam(), {{}}), incompatible_dimensions);
}

TEST_P(matrix_tests, costruct_with_dimensions) {
  matrix a(*GetParam(), 3U, 4U);
  EXPECT_EQ(3U, a.row_count());
  EXPECT_EQ(4U, a.column_count());
  for (size_t m = 0U; m < a.row_count(); ++m) {
    for (size_t n = 0U; n < a.column_count(); ++n) {
      EXPECT_EQ(0.0, a[m][n]) << "should be initialized to zero";
    }
  }
}

TEST_P(matrix_tests, costruct_with_initial_value) {
  const float initial_value(2014.0);
  matrix a(*GetParam(), 400U, 300U, initial_value);
  EXPECT_EQ(400U, a.row_count());
  EXPECT_EQ(300U, a.column_count());
  for (size_t m = 0U; m < a.row_count(); ++m) {
    for (size_t n = 0U; n < a.column_count(); ++n) {
      EXPECT_EQ(initial_value, a[m][n]) << "should be initialized to " << initial_value;
    }
  }
}

TEST_P(matrix_tests, costruct_with_initial_values_array) {
  float initial_values[] = {1.0, 2.0, 3.0, 4.0};
  std::shared_ptr<float> buffer(initial_values, [](float* ptr) { (void)ptr; });
  matrix a(*GetParam(), 2U, 2U, buffer);
  EXPECT_EQ(2U, a.row_count());
  EXPECT_EQ(2U, a.column_count());
  EXPECT_FLOAT_EQ(1.0, a[0][0]);
  EXPECT_FLOAT_EQ(2.0, a[0][1]);
  EXPECT_FLOAT_EQ(3.0, a[1][0]);
  EXPECT_FLOAT_EQ(4.0, a[1][1]);
}

TEST_P(matrix_tests, costruct_with_initializer_list) {
  matrix a(*GetParam(), {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  EXPECT_EQ(2U, a.row_count());
  EXPECT_EQ(3U, a.column_count());
  EXPECT_FLOAT_EQ(1.0, a[0][0]);
  EXPECT_FLOAT_EQ(2.0, a[0][1]);
  EXPECT_FLOAT_EQ(3.0, a[0][2]);
  EXPECT_FLOAT_EQ(4.0, a[1][0]);
  EXPECT_FLOAT_EQ(5.0, a[1][1]);
  EXPECT_FLOAT_EQ(6.0, a[1][2]);

  matrix b(*GetParam(), {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
  EXPECT_EQ(3U, b.row_count());
  EXPECT_EQ(2U, b.column_count());
  EXPECT_FLOAT_EQ(1.0, b[0][0]);
  EXPECT_FLOAT_EQ(2.0, b[0][1]);
  EXPECT_FLOAT_EQ(3.0, b[1][0]);
  EXPECT_FLOAT_EQ(4.0, b[1][1]);
  EXPECT_FLOAT_EQ(5.0, b[2][0]);
  EXPECT_FLOAT_EQ(6.0, b[2][1]);
}

TEST_P(matrix_tests, construct_with_size) {
  matrix a(*GetParam(), make_pair(42U, 12U));
  EXPECT_EQ(42U, a.row_count());
  EXPECT_EQ(12U, a.column_count());
}

TEST_P(matrix_tests, copy_construction) {
  const matrix a(*GetParam(), {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
  matrix b(a);
  EXPECT_MATRIX_EQ(a, b);

  float current = a[0][0];
  a[0][0] = current + 1.0;
  EXPECT_EQ(current + 1.0, b[0][0])
      << "copied matrix should change when the matrix it refers to changes";
}

TEST_P(matrix_tests, assignment) {
  const matrix a(*GetParam(), {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
  matrix b(*GetParam(), 1U, 1U);
  b = a;
  EXPECT_MATRIX_EQ(a, b);

  float current = a[0][0];
  a[0][0] = current + 1.0;
  EXPECT_EQ(current + 1.0, b[0][0])
      << "assigned matrix should change when the matrix it refers to changes";
}

TEST_P(matrix_tests, clone) {
  const matrix a(*GetParam(), {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
  matrix b(*GetParam(), 1U, 1U);
  b = a.clone();
  EXPECT_MATRIX_EQ(a, b);

  float current = a[0][0];
  a[0][0] = current + 1.0;
  EXPECT_EQ(current, b[0][0]) << "assigned matrix should not change when the "
                                 "matrix it was cloned from changes";
}

TEST_P(matrix_tests, access_valid_row_index) {
  matrix a(*GetParam(), 3U, 3U, 42.0);
  EXPECT_NO_THROW((void)a[0][0]);
  EXPECT_NO_THROW((void)a[1][1]);
  EXPECT_NO_THROW((void)a[2][2]);
}

TEST_P(matrix_tests, access_invalid_row) {
  matrix a(*GetParam(), 3U, 3U);
  EXPECT_THROW(a[3], std::out_of_range);
  EXPECT_THROW(a[42], std::out_of_range);
}

TEST_P(matrix_tests, matrix_matrix_multiplication) {
  matrix a(*GetParam(), {{1.0, 2.0}, {3.0, 4.0}});
  matrix b(*GetParam(), {{1.0, 2.0}, {3.0, 4.0}});
  matrix c = a * b;
  matrix expected(*GetParam(), {{7.0, 10.0}, {15.0, 22.0}});
  EXPECT_MATRIX_EQ(expected, c);
}

TEST_P(matrix_tests, matrix_matrix_multiplication_with_invalid_dimensions) {
  matrix a(*GetParam(), 4U, 3U);
  matrix b(*GetParam(), 4U, 2U);
  EXPECT_THROW(a * b, incompatible_dimensions);
}

TEST_P(matrix_tests, matrix_scalar_multiplication) {
  matrix a(*GetParam(), {{1.0, 2.0}, {3.0, 4.0}});
  matrix b(a * 3.0);
  matrix expected(*GetParam(), {{3.0, 6.0}, {9.0, 12.0}});
  EXPECT_MATRIX_EQ(expected, b);
}

TEST_P(matrix_tests, matrix_matrix_addition) {
  matrix a(*GetParam(), 2U, 2U, 1.0);
  matrix b(*GetParam(), 2U, 2U, 2.0);
  matrix c = a + b;
  matrix expected(*GetParam(), 2U, 2U, 3.0);
  EXPECT_MATRIX_EQ(expected, c);

  matrix d(*GetParam(), a.row_count() + 1U, a.column_count());
  EXPECT_NO_THROW(d + d);
  EXPECT_THROW(a + d, incompatible_dimensions);
  EXPECT_THROW(d + a, incompatible_dimensions);

  matrix e(*GetParam(), a.row_count(), a.column_count() + 1U);
  EXPECT_NO_THROW(e + e);
  EXPECT_THROW(a + e, incompatible_dimensions);
  EXPECT_THROW(e + a, incompatible_dimensions);
}

TEST_P(matrix_tests, matrix_scalar_addition) {
  matrix a(*GetParam(), 2U, 2U, 1.0);
  matrix b = a + 2.0;
  matrix expected(*GetParam(), 2U, 2U, 3.0);
  EXPECT_MATRIX_EQ(expected, b);
}

TEST_P(matrix_tests, matrix_matrix_subtraction) {
  matrix a(*GetParam(), 2U, 2U, 1.0);
  matrix b(*GetParam(), 2U, 2U, 2.0);
  matrix c = b - a;
  matrix expected(*GetParam(), 2U, 2U, 1.0);
  EXPECT_MATRIX_EQ(expected, c);

  matrix d(*GetParam(), a.row_count() + 1U, a.column_count());
  EXPECT_NO_THROW(d - d);
  EXPECT_THROW(a - d, incompatible_dimensions);
  EXPECT_THROW(d - a, incompatible_dimensions);

  matrix e(*GetParam(), a.row_count(), a.column_count() + 1U);
  EXPECT_NO_THROW(e - e);
  EXPECT_THROW(a - e, incompatible_dimensions);
  EXPECT_THROW(e - a, incompatible_dimensions);
}

TEST_P(matrix_tests, matrix_scalar_subtraction) {
  matrix a(*GetParam(), 2U, 2U, 4.0);
  matrix b = a - 2.0;
  matrix expected(*GetParam(), 2U, 2U, 2.0);
  EXPECT_MATRIX_EQ(expected, b);
}

TEST_P(matrix_tests, elementwise_product) {
  matrix a(*GetParam(), {
                            {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0},
                        });
  matrix b(*GetParam(), {
                            {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0},
                        });
  matrix c = a.elementwise_product(b);
  matrix expected(*GetParam(), {
                                   {1.0, 4.0, 9.0}, {16.0, 25.0, 36.0}, {49.0, 64.0, 81.0},
                               });
  EXPECT_MATRIX_EQ(expected, c);

  matrix d(*GetParam(), a.row_count() + 1U, a.column_count());
  EXPECT_THROW(a.elementwise_product(d), incompatible_dimensions);
  EXPECT_THROW(d.elementwise_product(a), incompatible_dimensions);

  matrix e(*GetParam(), a.row_count(), a.column_count() + 1U);
  EXPECT_THROW(a.elementwise_product(e), incompatible_dimensions);
  EXPECT_THROW(e.elementwise_product(a), incompatible_dimensions);

  matrix f(*GetParam(), a.row_count() + 1U, a.column_count() + 1U);
  EXPECT_THROW(a.elementwise_product(f), incompatible_dimensions);
  EXPECT_THROW(f.elementwise_product(a), incompatible_dimensions);
}

TEST_P(matrix_tests, transpose) {
  vi::la::matrix a(*GetParam(), 4U, 3U, 1.0);
  vi::la::matrix b(a.transpose());
  EXPECT_EQ(a.row_count(), b.column_count());
  EXPECT_EQ(a.column_count(), b.row_count());

  for (size_t m = 0U; m < b.row_count(); ++m) {
    for (size_t n = 0U; n < b.column_count(); ++n) {
      EXPECT_EQ(a[n][m], b[m][n]);
    }
  }
}

TEST_P(matrix_tests, merge) {
  matrix a(*GetParam(), 4U, 3U, 1.0);
  matrix b(*GetParam(), 4U, 2U, 2.0);
  matrix c = a << b;
  matrix expected(*GetParam(), {{1.0, 1.0, 1.0, 2.0, 2.0},
                                {1.0, 1.0, 1.0, 2.0, 2.0},
                                {1.0, 1.0, 1.0, 2.0, 2.0},
                                {1.0, 1.0, 1.0, 2.0, 2.0}});
  EXPECT_MATRIX_EQ(expected, c);
}

TEST_P(matrix_tests, sum_rows) {
  matrix a(*GetParam(), {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  matrix summed = GetParam()->sum_rows(a);
  matrix expected(*GetParam(), {{5.0, 7.0, 9.0}});
  EXPECT_MATRIX_EQ(expected, summed);
}

TEST_P(matrix_tests, sum_columns) {
  matrix a(*GetParam(), {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
  matrix summed = GetParam()->sum_columns(a);
  matrix expected(*GetParam(), {{6.0}, {15.0}});
}

TEST_P(matrix_tests, splice_columns_with_invalid_indices) {
  matrix a(*GetParam(), 3U, 5U);
  EXPECT_THROW(a.columns(1U, a.column_count()), std::out_of_range);
  EXPECT_THROW(a.columns(2U, 1U), std::out_of_range);
  EXPECT_THROW(a.columns(a.column_count(), a.column_count()), std::out_of_range);
}

TEST_P(matrix_tests, splice_columns) {
  matrix x(*GetParam(), {{1.0, 2.0}, {3.0, 4.0}});
  matrix x_0 = x.column(0U);
  matrix x_0_expected(*GetParam(), {{1.0}, {3.0}});
  EXPECT_MATRIX_EQ(x_0_expected, x_0);

  matrix x_1 = x.column(1U);
  matrix x_1_expected(*GetParam(), {{2.0}, {4.0}});
  EXPECT_MATRIX_EQ(x_1_expected, x_1);
}

TEST_P(matrix_tests, splice_rows_with_invalid_indices) {
  matrix a(*GetParam(), 5U, 4U);
  EXPECT_THROW(a.rows(1U, a.row_count()), std::out_of_range);
  EXPECT_THROW(a.rows(2U, 1U), std::out_of_range);
  EXPECT_THROW(a.rows(a.row_count(), a.row_count()), std::out_of_range);
}

TEST_P(matrix_tests, splice_rows) {
  matrix a(*GetParam(), 5U, 4U);
  matrix b(a.rows(0, 1));
  EXPECT_EQ(2U, b.row_count());
  EXPECT_EQ(4U, b.column_count());

  matrix c(a.row(1U));
  EXPECT_EQ(1U, c.row_count());
  EXPECT_EQ(4U, c.column_count());

  matrix x(*GetParam(), {{1.0, 2.0}, {3.0, 4.0}});
  matrix x_0 = x.row(0U);
  matrix x_0_expected(*GetParam(), {{1.0, 2.0}});
  EXPECT_MATRIX_EQ(x_0_expected, x_0);

  matrix x_1 = x.row(1U);
  matrix x_1_expected(*GetParam(), {{3.0, 4.0}});
  EXPECT_MATRIX_EQ(x_1, x_1_expected);
}

TEST_P(matrix_tests, sub_matrix_succeeds) {
  matrix a(*GetParam(), {
                            {1.0, 1.0, 2.0, 2.0},
                            {1.0, 1.0, 2.0, 2.0},

                            {3.0, 3.0, 4.0, 4.0},
                            {3.0, 3.0, 4.0, 4.0},
                        });

  matrix ones = a.sub_matrix(0, 1, 0, 1);
  EXPECT_MATRIX_EQ(matrix(*GetParam(), 2U, 2U, 1.0), ones);
  matrix twos = a.sub_matrix(0, 1, 2, 3);
  EXPECT_MATRIX_EQ(matrix(*GetParam(), 2U, 2U, 2.0), twos);
  matrix threes = a.sub_matrix(2, 3, 0, 1);
  EXPECT_MATRIX_EQ(matrix(*GetParam(), 2U, 2U, 3.0), threes);
  matrix fours = a.sub_matrix(2, 3, 2, 3);
  EXPECT_MATRIX_EQ(matrix(*GetParam(), 2U, 2U, 4.0), fours);
}

TEST_P(matrix_tests, sub_matrix_fails) {
  matrix a(*GetParam(), 6U, 6U);
  EXPECT_THROW(a.sub_matrix(0, a.row_count(), 0, 1), std::out_of_range);
  EXPECT_THROW(a.sub_matrix(0, 1, 0, a.column_count()), std::out_of_range);
  EXPECT_THROW(a.sub_matrix(0, a.row_count(), 0, a.column_count()), std::out_of_range);
}

// Convolutions are WIP - not working on OSX CPU based OpenCL due to missing
// 2-D kernel support
class DISABLED_matrix_convolution_tests : public ::testing::TestWithParam<vi::la::context*> {};
TEST_P(DISABLED_matrix_convolution_tests, convolve_2d_odd_mask) {
  matrix a(*GetParam(), 4U, 4U, 1.0f);
  matrix mask(*GetParam(), {{1.0, 1.0, 1.0}, {1.0, 2.0, 1.0}, {1.0, 1.0, 1.0}});
  matrix expected(
      *GetParam(),
      {{5.0, 7.0, 7.0, 5.0}, {7.0, 10.0, 10.0, 7.0}, {7.0, 10.0, 10.0, 7.0}, {5.0, 7.0, 7.0, 5.0}});
  matrix result(*GetParam(), a.size(), 0.0f);
  GetParam()->convolve_2d(result, mask, a, 1);
  EXPECT_MATRIX_EQ(expected, result);
}

TEST_P(DISABLED_matrix_convolution_tests, convolve_2d_2channels_odd_mask) {
  matrix a(*GetParam(), {{1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 10.0, 20.0, 10.0, 20.0},
                         {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 10.0, 20.0, 10.0, 20.0},
                         {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 10.0, 20.0, 10.0, 20.0},
                         {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 10.0, 20.0, 10.0, 20.0}});
  matrix mask(*GetParam(), {{1.0, 1.0, 1.0}, {1.0, 2.0, 1.0}, {1.0, 1.0, 1.0}});
  matrix expected(*GetParam(),
                  {{5.0, 10.0, 7.0, 14.0, 7.0, 14.0, 25.0, 50.0, 52.0, 104.0, 50.0, 100.0},
                   {7.0, 14.0, 10.0, 20.0, 10.0, 20.0, 37.0, 74.0, 73.0, 146.0, 70.0, 140.0},
                   {7.0, 14.0, 10.0, 20.0, 10.0, 20.0, 37.0, 74.0, 73.0, 146.0, 70.0, 140.0},
                   {5.0, 10.0, 7.0, 14.0, 7.0, 14.0, 25.0, 50.0, 52.0, 104.0, 50.0, 100.0}});
  matrix result(*GetParam(), a.size(), 0.0f);
  GetParam()->convolve_2d(result, mask, a, 2);
  EXPECT_MATRIX_EQ(expected, result);
}

TEST_P(DISABLED_matrix_convolution_tests, convolve_2d_even_rows_mask) {
  matrix a(*GetParam(), 4U, 4U, 1.0f);
  matrix mask(*GetParam(), {{1.0, 2.0, 1.0}, {1.0, 2.0, 1.0}});
  matrix expected(
      *GetParam(),
      {{3.0, 4.0, 4.0, 3.0}, {6.0, 8.0, 8.0, 6.0}, {6.0, 8.0, 8.0, 6.0}, {6.0, 8.0, 8.0, 6.0}});

  matrix result(*GetParam(), a.size(), 0.0f);
  GetParam()->convolve_2d(result, mask, a, 1);
  EXPECT_MATRIX_EQ(expected, result);
}

// WIP, even column mask is broken
TEST_P(DISABLED_matrix_convolution_tests, convolve_2d_even_colums_mask) {
  matrix a(*GetParam(), 4U, 4U, 1.0f);
  matrix mask(*GetParam(), {{1.0, 2.0}, {1.0, 2.0}, {1.0, 2.0}});
  matrix expected(
      *GetParam(),
      {{4.0, 6.0, 6.0, 6.0}, {6.0, 9.0, 9.0, 9.0}, {6.0, 9.0, 9.0, 9.0}, {4.0, 6.0, 6.0, 6.0}});
  matrix result(*GetParam(), a.size(), 0.0f);
  GetParam()->convolve_2d(result, mask, a, 1);
  cout << result << endl;
  EXPECT_MATRIX_EQ(expected, result);
}

TEST_P(DISABLED_matrix_convolution_tests, convolve_2d_even_mask) {
  matrix a(*GetParam(), 4U, 4U, 1.0f);
  matrix mask(*GetParam(), {{1.0, 2.0}, {1.0, 2.0}});
  matrix expected(
      *GetParam(),
      {{2.0, 3.0, 3.0, 3.0}, {4.0, 6.0, 6.0, 6.0}, {4.0, 6.0, 6.0, 6.0}, {4.0, 6.0, 6.0, 6.0}});
  matrix result(*GetParam(), a.size(), 0.0f);
  GetParam()->convolve_2d(result, mask, a, 1);
  EXPECT_MATRIX_EQ(expected, result);
}
