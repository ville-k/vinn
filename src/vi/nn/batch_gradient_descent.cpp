#include "vi/nn/batch_gradient_descent.h"
#include "vi/nn/cost_function.h"
#include "vi/nn/layer.h"

#include <cassert>
#include <algorithm>
#include <iostream>

namespace vi {
namespace nn {

batch_gradient_descent::batch_gradient_descent(size_t max_epoch_count,
                                               double learning_rate)
    : _max_epoch_count(max_epoch_count), _learning_rate(learning_rate) {
  assert(_max_epoch_count > 0U);
  assert(_learning_rate > 0.0);
}

double batch_gradient_descent::train(vi::nn::network& network,
                                     const vi::la::matrix& features,
                                     const vi::la::matrix& targets,
                                     vi::nn::cost_function& cost_function) {
  double cost(std::numeric_limits<double>::max());
  for (size_t epoch = 1U; epoch <= _max_epoch_count; ++epoch) {
    std::pair<double, std::vector<vi::la::matrix>> cost_and_gradients =
        network.backward(features, targets, cost_function);
    std::vector<vi::la::matrix>& gradients = cost_and_gradients.second;
    for (size_t i = 0U; i < gradients.size(); ++i) {
      layer& l = network.layers()[i];
      vi::la::matrix theta = l.get_theta() - (gradients[i] * _learning_rate);
      l.set_theta(theta);
    }

    cost = cost_and_gradients.first;
    if (_stop_early && _stop_early(network, epoch, cost)) {
      return cost;
    }
  }

  return cost;
}

}
}

