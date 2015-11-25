#include "test.h"
#include "vi/nn/layer.h"
#include "vi/nn/activation_function.h"
#include "vi/nn/cost_function.h"
#include "vi/nn/l2_regularizer.h"
#include "vi/nn/network.h"
#include "vi/nn/trainer.h"

class trainer_tests
    : public ::testing::TestWithParam<testing::tuple<vi::la::context*, vi::nn::trainer*>> {

public:
  trainer_tests() : _l2_regularizer(0.5) {}

protected:
  virtual void SetUp() {
    vi::la::context* context = std::get<0>(GetParam());
    _trainer = std::get<1>(GetParam());

    _network = new vi::nn::network;
    _network->add(std::make_shared<vi::nn::layer>(
        *context, std::make_shared<vi::nn::sigmoid_activation>(), 25, 20 * 20));
    _network->add(std::make_shared<vi::nn::layer>(
        *context, std::make_shared<vi::nn::softmax_activation>(), 10, 25));

    _features = new vi::la::matrix(*context, 100U, 400U, 42.0f);
    _targets = new vi::la::matrix(*context, 100U, 10U, 0.0f);
    for (size_t row = 0U; row < 100U; ++row) {
      (*_targets)[row][0] = 1.0;
    }
    _max_epochs = 5U;
  }

  virtual void TearDown() {
    delete _network;
    delete _features;
    delete _targets;
  }

  vi::nn::trainer* _trainer;
  vi::nn::network* _network;
  vi::nn::cross_entropy_cost _cost_function;
  vi::nn::l2_regularizer _l2_regularizer;
  vi::la::matrix* _features;
  vi::la::matrix* _targets;
  size_t _max_epochs;
};

TEST_P(trainer_tests, train_succeeds) {
  float final_cost = _trainer->train(*_network, *_features, *_targets, _cost_function);
  EXPECT_LT(0.0f, final_cost);
}

TEST_P(trainer_tests, train_with_l2_regularizer_succeeds) {
  float final_cost =
      _trainer->train(*_network, *_features, *_targets, _cost_function, _l2_regularizer);
  EXPECT_LT(0.0f, final_cost);
}

TEST_P(trainer_tests, train_calls_early_stopping_callback) {
  size_t early_stopping_called(0U);
  _trainer->set_stop_early([&early_stopping_called](const vi::nn::network&, size_t, float) -> bool {
    early_stopping_called++;
    return false;
  });

  float final_cost = _trainer->train(*_network, *_features, *_targets, _cost_function);
  EXPECT_LT(0.0f, final_cost);
  // softmax layer should never find a solution
  EXPECT_EQ(_max_epochs, early_stopping_called);
}

TEST_P(trainer_tests, train_stops_training_early) {
  size_t early_stopping_called(0U);
  _trainer->set_stop_early([&early_stopping_called](const vi::nn::network&, size_t, float) -> bool {
    early_stopping_called++;
    return true;
  });

  float final_cost = _trainer->train(*_network, *_features, *_targets, _cost_function);
  EXPECT_LT(0.0f, final_cost);
  EXPECT_EQ(1U, early_stopping_called);
}

#include "vi/nn/batch_gradient_descent.h"
#include "vi/nn/minibatch_gradient_descent.h"

INSTANTIATE_TEST_CASE_P(
    interface, trainer_tests,
    ::testing::Combine(::testing::ValuesIn(test::all_contexts()),
                       ::testing::Values(new vi::nn::minibatch_gradient_descent(5, 0.3f, 10),
                                         new vi::nn::batch_gradient_descent(5, 0.3f))));
