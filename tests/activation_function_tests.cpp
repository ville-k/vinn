#include "test.h"
#include "vi/nn/activation_function.h"

#include <iostream>

using namespace vi::la;
using namespace vi::nn;
using namespace std;

class activation_function_tests : public ::testing::TestWithParam<vi::la::context*> {};
INSTANTIATE_TEST_CASE_P(context, activation_function_tests,
                        ::testing::ValuesIn(test::all_contexts()));

TEST_P(activation_function_tests, sigmoid_activation) {
  sigmoid_activation activation;
  matrix input(*GetParam(), {{-1.0, -0.5}, {0.5, 1.0}});

  activation.activate(input);
  const float max_error = 0.0000001;
  EXPECT_NEAR(0.2689414, input[0][0], max_error);
  EXPECT_NEAR(0.3775406, input[0][1], max_error);
  EXPECT_NEAR(0.6224593, input[1][0], max_error);
  EXPECT_NEAR(0.7310586, input[1][1], max_error);
}

TEST_P(activation_function_tests, sigmoid_gradient) {
  matrix input(*GetParam(), {{-1.0, -0.5}, {0.5, 1.0}});

  sigmoid_activation activation;
  activation.activate(input);
  const matrix gradient = activation.gradient(input);
  EXPECT_EQ(input.size(), gradient.size());
  const float max_error = 0.0000001;
  EXPECT_NEAR(0.1966119234, gradient[0][0], max_error);
  EXPECT_NEAR(0.2350036954, gradient[0][1], max_error);
  EXPECT_NEAR(0.2350037198, gradient[1][0], max_error);
  EXPECT_NEAR(0.1966119234, gradient[1][1], max_error);
}

TEST_P(activation_function_tests, softmax_activation) {
  softmax_activation activation;
  matrix input(*GetParam(), {{-1.0, -0.5}, {0.5, 1.0}});

  activation.activate(input);
  const float max_error = 0.0000001;
  EXPECT_NEAR(0.3775406688, input[0][0], max_error);
  EXPECT_NEAR(0.6224593312, input[0][1], max_error);
  EXPECT_NEAR(0.3775406688, input[1][0], max_error);
  EXPECT_NEAR(0.6224593312, input[1][1], max_error);
}

// TEST_P(activation_function_tests, softmax_gradient) {
//  matrix input(*GetParam(), {{-1.0f, -0.5f},
//                             {0.5f,  1.0f}});
//
//  softmax_activation activation;
//  activation.activate(input);
//  const matrix gradient = activation.gradient(input);
//  EXPECT_EQ(input.size(), gradient.size());
//  EXPECT_FLOAT_EQ(0.4199743415, gradient[0][0]);
//  EXPECT_FLOAT_EQ(0.7864477329, gradient[0][1]);
//  EXPECT_FLOAT_EQ(0.7864477329, gradient[1][0]);
//  EXPECT_FLOAT_EQ(0.4199743415, gradient[1][1]);
//}

TEST_P(activation_function_tests, hyperbolic_tangent_activation) {
  hyperbolic_tangent activation;
  matrix input(*GetParam(), {{-1.0, -0.5}, {0.5, 1.0}});

  activation.activate(input);
  const float max_error = 0.0000001;
  EXPECT_NEAR(-0.761594156, input[0][0], max_error);
  EXPECT_NEAR(-0.462117157, input[0][1], max_error);
  EXPECT_NEAR(0.462117157, input[1][0], max_error);
  EXPECT_NEAR(0.761594156, input[1][1], max_error);
}

TEST_P(activation_function_tests, hyperbolic_tangent_gradient) {
  matrix input(*GetParam(), {{-1.0, -0.5}, {0.5, 1.0}});

  hyperbolic_tangent activation;
  activation.activate(input);
  const matrix gradient = activation.gradient(input);
  const float max_error = 0.0000001;
  EXPECT_NEAR(0.4199743415, gradient[0][0], max_error);
  EXPECT_NEAR(0.7864477329, gradient[0][1], max_error);
  EXPECT_NEAR(0.7864477329, gradient[1][0], max_error);
  EXPECT_NEAR(0.4199743415, gradient[1][1], max_error);
}
