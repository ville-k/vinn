#include "l2_regularizer.h"

namespace vi {
namespace nn {

l2_regularizer::l2_regularizer(float weight_decay) : _weight_decay(weight_decay) {}

std::pair<float, vi::la::matrix> l2_regularizer::penalty(const vi::la::matrix& weights) const {
  la::context& context = weights.owning_context();

  const vi::la::matrix zeros(context, weights.row_count(), 1U, 0.0);
  const vi::la::matrix biasless_weights =
      weights.sub_matrix(0U, weights.row_count() - 1U, 1U, weights.column_count() - 1U);

  const vi::la::matrix weights_squared = biasless_weights.elementwise_product(biasless_weights);
  const float cost_penalty =
      (_weight_decay / 2.0) * context.sum_columns(context.sum_rows(weights_squared))[0][0];

  const vi::la::matrix gradient_penalty = zeros << (biasless_weights * _weight_decay);

  return std::make_pair(cost_penalty, gradient_penalty);
}
}
}
