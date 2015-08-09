#include "test.h"
#include "vi/nn/activation_function.h"
#include "vi/nn/batch_gradient_descent.h"
#include "vi/nn/minibatch_gradient_descent.h"

#include "vi/nn/cost_function.h"
#include "vi/nn/network.h"

#include "vi/nn/label_map.h"
#include "vi/la/opencl/opencl_context.h"
#include "vi/nn/result_measurements.h"

class network_tests : public ::testing::TestWithParam<vi::la::context*> {};
INSTANTIATE_TEST_CASE_P(context, network_tests, ::testing::ValuesIn(test::all_contexts()));

using vi::la::matrix;
using vi::nn::network;
using vi::nn::layer;

TEST_P(network_tests, constructing_compatible_layers_succeeds) {
  layer l1(*GetParam(), new vi::nn::sigmoid_activation(), 25, 400);
  layer l2(*GetParam(), new vi::nn::softmax_activation(), 10, 25);
  EXPECT_NO_THROW(network(*GetParam(), {l1, l2}));
}

TEST_P(network_tests, constructing_incompatible_layers_fails) {
  layer l1(*GetParam(), new vi::nn::sigmoid_activation(), 42, 400);
  layer l2(*GetParam(), new vi::nn::softmax_activation(), 10, 25);
  EXPECT_THROW(network(*GetParam(), {l1, l2}), vi::nn::network::invalid_configuration);
}

TEST_P(network_tests, forward_single_example_succeeds) {
  layer l1(*GetParam(), new vi::nn::sigmoid_activation(), 25, 400);
  layer l2(*GetParam(), new vi::nn::softmax_activation(), 10, 25);
  network network(*GetParam(), {l1, l2});

  matrix inputs(*GetParam(), 1, 400U, 1.0);
  matrix predictions = network.forward(inputs);
  ASSERT_EQ(1U, predictions.row_count());
  ASSERT_EQ(10U, predictions.column_count());
}

TEST_P(network_tests, forward_multiple_examples_succeed) {
  layer l1(*GetParam(), new vi::nn::sigmoid_activation(), 25, 400);
  layer l2(*GetParam(), new vi::nn::softmax_activation(), 10, 25);
  network network(*GetParam(), {l1, l2});

  matrix inputs(*GetParam(), 42, 400U, 1.0);
  matrix predictions = network.forward(inputs);
  ASSERT_EQ(42U, predictions.row_count());
  ASSERT_EQ(10U, predictions.column_count());
}

TEST_P(network_tests, forward_invalid_dimensions_fails) {
  layer l1(*GetParam(), new vi::nn::sigmoid_activation(), 25, 400);
  layer l2(*GetParam(), new vi::nn::softmax_activation(), 10, 25);
  network network(*GetParam(), {l1, l2});

  matrix inputs(*GetParam(), 42, 7U, 1.0);
  EXPECT_THROW(network.forward(inputs), vi::la::incompatible_dimensions);
}

TEST_P(network_tests, backward_succeeds_with_valid_inputs) {
  layer l1(*GetParam(), new vi::nn::sigmoid_activation(), 25, 10);
  layer l2(*GetParam(), new vi::nn::softmax_activation(), 10, 25);
  network network(*GetParam(), {l1, l2});

  matrix features(*GetParam(), 10, 10);
  matrix targets(*GetParam(), 10, 10, 1.0);

  vi::nn::cross_entropy_cost cost_function;
  std::pair<float, std::vector<matrix>> cost_and_gradients =
      network.backward(features, targets, cost_function);

  ASSERT_LT(0.0, cost_and_gradients.first);
  ASSERT_EQ(2U, cost_and_gradients.second.size());
  ASSERT_EQ(l1.get_weights().size(), cost_and_gradients.second[0].size());
  ASSERT_EQ(l2.get_weights().size(), cost_and_gradients.second[1].size());
}

TEST_P(network_tests, backward_fails_with_invalid_feature_dimensions) {
  layer l1(*GetParam(), new vi::nn::sigmoid_activation(), 25, 10);
  layer l2(*GetParam(), new vi::nn::softmax_activation(), 10, 25);
  network network(*GetParam(), {l1, l2});

  matrix features(*GetParam(), 10, 7);
  matrix targets(*GetParam(), 10, 10, 1.0);

  vi::nn::cross_entropy_cost cost_function;
  EXPECT_THROW(network.backward(features, targets, cost_function), vi::la::incompatible_dimensions);
}

TEST_P(network_tests, backward_fails_with_different_number_features_and_targets) {
  layer l1(*GetParam(), new vi::nn::sigmoid_activation(), 25, 10);
  layer l2(*GetParam(), new vi::nn::softmax_activation(), 10, 25);
  network network(*GetParam(), {l1, l2});

  matrix features(*GetParam(), 10, 10);
  matrix targets(*GetParam(), 7, 10, 1.0);

  vi::nn::cross_entropy_cost cost_function;
  EXPECT_THROW(network.backward(features, targets, cost_function), vi::la::incompatible_dimensions);
}

TEST_P(network_tests, backward_fails_with_invalid_target_dimensions) {
  layer l1(*GetParam(), new vi::nn::sigmoid_activation(), 25, 10);
  layer l2(*GetParam(), new vi::nn::softmax_activation(), 10, 25);
  network network(*GetParam(), {l1, l2});

  matrix features(*GetParam(), 10, 10);
  matrix targets(*GetParam(), 10, 7, 1.0);

  vi::nn::cross_entropy_cost cost_function;
  EXPECT_THROW(network.backward(features, targets, cost_function), vi::la::incompatible_dimensions);
}
