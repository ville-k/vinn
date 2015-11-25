#include "vi/nn/minibatch_gradient_descent.h"
#include "vi/nn/batch_gradient_descent.h"
#include "vi/nn/cost_function.h"
#include "vi/nn/layer.h"
#include "vi/nn/running_average.h"

#include <cassert>
#include <algorithm>
#include <iostream>

namespace vi {
namespace nn {

minibatch_gradient_descent::minibatch_gradient_descent(const size_t max_epoch_count,
                                                       const float learning_rate,
                                                       const size_t batch_size,
                                                       const size_t batch_iteration_count)
    : _learning_rate(learning_rate), _max_epoch_count(max_epoch_count), _batch_size(batch_size),
      _batch_iteration_count(batch_iteration_count) {
  assert(_learning_rate > 0.0);
  assert(_max_epoch_count > 0U);
  assert(_batch_size > 0U);
  assert(_batch_iteration_count > 0U);
}

float minibatch_gradient_descent::train(vi::nn::network& network, const vi::la::matrix& features,
                                        const vi::la::matrix& targets,
                                        vi::nn::cost_function& cost_function) {
  return train(network, features, targets, cost_function, nullptr);
}

float minibatch_gradient_descent::train(vi::nn::network& network, const vi::la::matrix& features,
                                        const vi::la::matrix& targets,
                                        vi::nn::cost_function& cost_function,
                                        const vi::nn::l2_regularizer& regularizer) {
  return train(network, features, targets, cost_function, &regularizer);
}

float minibatch_gradient_descent::train(vi::nn::network& network, const vi::la::matrix& features,
                                        const vi::la::matrix& targets,
                                        vi::nn::cost_function& cost_function,
                                        const vi::nn::l2_regularizer* regularizer) {
  assert(_batch_size <= features.row_count());
  const size_t effective_batch_size = std::min(_batch_size, features.row_count());
  const size_t batch_count(features.row_count() / effective_batch_size);
  const size_t maximum_batches_to_average(20U);
  const size_t batches_to_average(std::min(maximum_batches_to_average, batch_count));
  running_average average(batches_to_average);

  for (size_t epoch = 1U; epoch <= _max_epoch_count; ++epoch) {
    batch_gradient_descent gd(_batch_iteration_count, _learning_rate);

    for (size_t batch = 0U; batch < batch_count; ++batch) {
      size_t batch_start = batch * effective_batch_size;
      size_t batch_end = batch_start + effective_batch_size - 1U;

      vi::la::matrix batch_features = features.rows(batch_start, batch_end);
      vi::la::matrix batch_targets = targets.rows(batch_start, batch_end);

      float batch_cost(0.0);
      if (regularizer) {
        batch_cost = gd.train(network, batch_features, batch_targets, cost_function, *regularizer);
      } else {
        batch_cost = gd.train(network, batch_features, batch_targets, cost_function);
      }
      average.add_value(batch_cost);
    }

    const float current_average = average.calculate();
    if (_stop_early && _stop_early(network, epoch, current_average)) {
      return current_average;
    }
  }

  return average.calculate();
}
}
}
