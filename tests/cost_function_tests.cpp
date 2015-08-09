#include "test.h"
#include "vi/nn/cost_function.h"

#include <iostream>

using namespace std;
using namespace vi::la;
using namespace vi::nn;

class cost_function_tests : public ::testing::TestWithParam<vi::la::context*> {};

INSTANTIATE_TEST_CASE_P(context, cost_function_tests, ::testing::ValuesIn(test::all_contexts()));

TEST_P(cost_function_tests, calculates_squared_error_cost) {
  matrix expected(*GetParam(), {{1.0, 2.0}, {3.0, 4.0}});
  matrix actual(*GetParam(), {{3.0, 0.0}, {1.0, 6.0}});
  squared_error_cost squared_error;
  matrix cost = squared_error.cost(expected, actual);

  const matrix correct(*GetParam(), {{4.0}, {4.0}});
  EXPECT_MATRIX_EQ(correct, cost);
}

TEST_P(cost_function_tests, calculates_squared_error_cost_derivative) {
  const matrix expected(*GetParam(), {{1.0, 2.0}, {3.0, 4.0}});
  const matrix actual(*GetParam(), {{3.0, 0.0}, {1.0, 6.0}});
  const matrix correct(*GetParam(), {{-2.0, 2.0}, {2.0, -2.0}});
  squared_error_cost squared_error;
  matrix cost_derivative = squared_error.cost_derivative(expected, actual);
  EXPECT_MATRIX_EQ(correct, cost_derivative);
}

TEST_P(cost_function_tests, calculates_cross_entropy_cost) {
  const matrix expected(*GetParam(), {{1.0, 0.0}, {0.0, 1.0}});
  const matrix actual(*GetParam(), {{.90, 0.10}, {.90, 0.10}});
  const matrix correct_cost(*GetParam(), {{0.10536}, {2.30259}});
  const float max_error = 0.00001;

  cross_entropy_cost cross_entropy;
  matrix cost = cross_entropy.cost(expected, actual);

  EXPECT_FLOAT_EQ(correct_cost.row_count(), cost.row_count());
  EXPECT_FLOAT_EQ(correct_cost.column_count(), cost.column_count());
  for (size_t m = 0U; m < correct_cost.row_count(); ++m) {
    for (size_t n = 0U; n < correct_cost.column_count(); ++n) {
      EXPECT_NEAR(correct_cost[m][n], cost[m][n], max_error);
    }
  }
}

TEST_P(cost_function_tests, calculates_cross_entropy_cost_derivative) {
  const matrix expected(*GetParam(), {{1.0, 0.0}, {0.0, 1.0}});
  const matrix actual(*GetParam(), {{.90, 0.10}, {.90, 0.10}});
  const matrix correct_derivative(*GetParam(), {{-0.1, 0.1}, {0.9, -0.9}});
  const float max_error = 0.00001;

  cross_entropy_cost cross_entropy;
  matrix cost_derivative = cross_entropy.cost_derivative(expected, actual);
  EXPECT_FLOAT_EQ(correct_derivative.row_count(), cost_derivative.row_count());
  EXPECT_FLOAT_EQ(correct_derivative.column_count(), cost_derivative.column_count());
  for (size_t m = 0U; m < correct_derivative.row_count(); ++m) {
    for (size_t n = 0U; n < correct_derivative.column_count(); ++n) {
      EXPECT_NEAR(correct_derivative[m][n], cost_derivative[m][n], max_error);
    }
  }
}
