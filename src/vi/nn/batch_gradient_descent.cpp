#include "vi/nn/batch_gradient_descent.h"
#include "vi/nn/cost_function.h"
#include "vi/nn/layer.h"
#include "vi/nn/l2_regularizer.h"
#include <cassert>
#include <algorithm>
#include <iostream>

namespace vi {
namespace nn {

batch_gradient_descent::batch_gradient_descent(size_t max_epoch_count, float learning_rate)
    : _max_epoch_count(max_epoch_count), _learning_rate(learning_rate) {
  assert(_max_epoch_count > 0U);
  assert(_learning_rate > 0.0);
}

float batch_gradient_descent::train(vi::nn::network& network, const vi::la::matrix& features,
                                    const vi::la::matrix& targets,
                                    vi::nn::cost_function& cost_function) {
  return train(network, features, targets, cost_function, nullptr);
}

float batch_gradient_descent::train(vi::nn::network& network, const vi::la::matrix& features,
                                    const vi::la::matrix& targets,
                                    vi::nn::cost_function& cost_function,
                                    const vi::nn::l2_regularizer& regularizer) {
  return train(network, features, targets, cost_function, &regularizer);
}

float batch_gradient_descent::train(vi::nn::network& network, const vi::la::matrix& features,
                                    const vi::la::matrix& targets,
                                    vi::nn::cost_function& cost_function,
                                    const vi::nn::l2_regularizer* regularizer) {
  float cost(std::numeric_limits<float>::max());
  const size_t example_count = features.row_count();

  for (size_t epoch = 1U; epoch <= _max_epoch_count; ++epoch) {
    std::pair<float, std::vector<vi::la::matrix>> cost_and_gradients =
        network.backward(features, targets, cost_function);

    cost = cost_and_gradients.first / example_count;
    std::vector<vi::la::matrix>& gradients = cost_and_gradients.second;

    size_t layer_index = 0U;
    for (std::shared_ptr<layer> l : network) {
      vi::la::matrix gradient = gradients[layer_index] / example_count;

      if (regularizer) {
        std::pair<float, vi::la::matrix> cost_and_gradient_penalty =
            regularizer->penalty(l->weights());
        cost += cost_and_gradient_penalty.first / example_count;
        gradient = gradient + (cost_and_gradient_penalty.second / example_count);
      }

      vi::la::matrix new_weights = l->weights() - (gradient * _learning_rate);
      l->weights(new_weights);

      ++layer_index;
    }

    if (_stop_early && _stop_early(network, epoch, cost)) {
      break;
    }
  }

  return cost;
}
}
}
