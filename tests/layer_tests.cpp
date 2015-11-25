#include "test.h"
#include "vi/nn/activation_function.h"
#include "vi/nn/cost_function.h"
#include "vi/nn/layer.h"

#include <iostream>

using namespace std;
using namespace vi::la;
using namespace vi::nn;

class layer_tests : public ::testing::TestWithParam<vi::la::context*> {

protected:
  matrix calculate_numerical_gradient(const layer& l, cost_function& cost_function,
                                      const matrix& input, const matrix& expected_output, float e) {
    matrix perturbed_weights(l.weights());
    matrix numerical_gradient(*GetParam(), l.weights().size(), 0.0);
    for (size_t m = 0U; m < perturbed_weights.row_count(); ++m) {
      for (size_t n = 0U; n < perturbed_weights.column_count(); ++n) {
        float original = perturbed_weights[m][n];

        perturbed_weights[m][n] = original - e;
        layer pl1(l);
        pl1.weights(perturbed_weights);

        auto hypothesis = pl1.forward(input);
        matrix cost1 = cost_function.cost(expected_output, hypothesis);
        float loss1 = cost1[0][0];

        perturbed_weights[m][n] = original + e;
        layer pl2(l);
        pl2.weights(perturbed_weights);

        auto hypothesis2 = pl2.forward(input);
        matrix cost2 = cost_function.cost(expected_output, hypothesis2);
        float loss2 = cost2[0][0];

        numerical_gradient[m][n] = (loss2 - loss1) / (2.0 * e);

        perturbed_weights[m][n] = original;
      }
    }
    return numerical_gradient;
  }

  float random(float start_range, float end_range) {
    float value(((float)std::rand()) / ((float)RAND_MAX));
    float range(end_range - start_range);
    return start_range + value * range;
  }

  void randomize(matrix& matrix, float start = 0.0, float end = 1.0) {
    for (size_t m = 0U; m < matrix.row_count(); ++m) {
      for (size_t n = 0U; n < matrix.column_count(); ++n) {
        matrix[m][n] = random(start, end);
      }
    }
  }
};
INSTANTIATE_TEST_CASE_P(context, layer_tests, ::testing::ValuesIn(test::all_contexts()));

TEST_P(layer_tests, forward_with_single_example) {
  const size_t input_units = 5U;
  const size_t output_units = 3U;
  vi::nn::layer l(*GetParam(), std::make_shared<vi::nn::sigmoid_activation>(), output_units,
                  input_units);

  vi::la::matrix input(*GetParam(), 1, input_units, 1.0);
  vi::la::matrix output = l.forward(input);
  EXPECT_EQ(1U, output.row_count());
  EXPECT_EQ(output_units, output.column_count());
}

TEST_P(layer_tests, forward_with_multiple_examples) {
  const size_t input_units = 5U;
  const size_t output_units = 3U;
  vi::nn::layer l(*GetParam(), std::make_shared<sigmoid_activation>(), output_units, input_units);

  vi::la::matrix input(*GetParam(), 3, input_units, 1.0);
  vi::la::matrix output = l.forward(input);
  EXPECT_EQ(3U, output.row_count());
  EXPECT_EQ(output_units, output.column_count());
}

TEST_P(layer_tests, backward) {
  const size_t input_units = 5U;
  const size_t output_units = 3U;
  layer l(*GetParam(), std::make_shared<sigmoid_activation>(), output_units, input_units);

  matrix input(*GetParam(), 1U, input_units, 1U);
  matrix activations = l.forward(input);
  matrix next_error(*GetParam(), 1U, output_units, 2.0);
  std::pair<matrix, matrix> delta_and_gradient = l.backward(input, activations, next_error);

  matrix& delta(delta_and_gradient.first);
  EXPECT_EQ(input.size(), delta.size());
  matrix& gradient(delta_and_gradient.second);
  EXPECT_EQ(l.weights().size(), gradient.size());
}

TEST_P(layer_tests, gradient_check_sigmoid_activation) {
  srand(0U);
  const size_t input_units(2U);
  const size_t output_units(6U);
  layer l(*GetParam(), std::make_shared<sigmoid_activation>(), output_units, input_units);

  matrix input(*GetParam(), 1U, input_units, 1U);
  randomize(input);
  matrix expected_output(*GetParam(), 1U, output_units, 1.0);
  randomize(expected_output);

  squared_error_cost cost_function;
  matrix activations = l.forward(input);
  matrix next_error = cost_function.cost_derivative(expected_output, activations);

  std::pair<matrix, matrix> delta_and_gradient = l.backward(input, activations, next_error);
  const float e = 0.01;
  const float max_error = 0.001;
  const matrix numerical_gradient =
      calculate_numerical_gradient(l, cost_function, input, expected_output, e);
  const matrix& gradient(delta_and_gradient.second);
  EXPECT_EQ(numerical_gradient.size(), gradient.size());
  for (size_t m = 0U; m < numerical_gradient.row_count(); ++m) {
    for (size_t n = 0U; n < numerical_gradient.column_count(); ++n) {
      EXPECT_NEAR(numerical_gradient[m][n], gradient[m][n], max_error);
    }
  }
}

TEST_P(layer_tests, gradient_check_hyperbolic_tangent_activation) {
  srand(0U);
  const size_t input_units(2U);
  const size_t output_units(6U);
  layer l(*GetParam(), std::make_shared<hyperbolic_tangent>(), output_units, input_units);

  matrix input(*GetParam(), 1U, input_units, 1U);
  randomize(input);
  matrix expected_output(*GetParam(), 1U, output_units, 1.0);
  randomize(expected_output);

  squared_error_cost cost_function;
  matrix activations = l.forward(input);

  matrix next_error = cost_function.cost_derivative(expected_output, activations);

  std::pair<matrix, matrix> delta_and_gradient = l.backward(input, activations, next_error);

  const float e = 0.01;
  const float max_error = 0.001;
  const matrix numerical_gradient =
      calculate_numerical_gradient(l, cost_function, input, expected_output, e);
  const matrix& gradient(delta_and_gradient.second);
  EXPECT_EQ(numerical_gradient.size(), gradient.size());
  for (size_t m = 0U; m < numerical_gradient.row_count(); ++m) {
    for (size_t n = 0U; n < numerical_gradient.column_count(); ++n) {
      EXPECT_NEAR(numerical_gradient[m][n], gradient[m][n], max_error);
    }
  }
}

TEST_P(layer_tests, gradient_check_softmax_activation) {
  srand(0U);
  const size_t input_units(2U);
  const size_t output_units(6U);
  layer l(*GetParam(), std::make_shared<softmax_activation>(), output_units, input_units);

  matrix input(*GetParam(), 1U, input_units, 1U);
  randomize(input);
  matrix expected_output(*GetParam(), 1U, output_units, 0.0);
  expected_output[0][0] = 1.0;

  matrix activations = l.forward(input);
  cross_entropy_cost cost_function;
  matrix next_error = cost_function.cost_derivative(expected_output, activations);
  std::pair<matrix, matrix> delta_and_gradient = l.backward(input, activations, next_error);

  const float e = 0.01;
  const float max_error = 0.001;
  const matrix numerical_gradient =
      calculate_numerical_gradient(l, cost_function, input, expected_output, e);

  const matrix& delta(delta_and_gradient.first);
  const matrix& gradient(delta_and_gradient.second);
  EXPECT_EQ(numerical_gradient.size(), gradient.size());
  EXPECT_EQ(input.size(), delta.size());
  for (size_t m = 0U; m < numerical_gradient.row_count(); ++m) {
    for (size_t n = 0U; n < numerical_gradient.column_count(); ++n) {
      EXPECT_NEAR(numerical_gradient[m][n], gradient[m][n], max_error);
    }
  }
}
