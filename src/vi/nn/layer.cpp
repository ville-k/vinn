#include "vi/nn/layer.h"
#include "vi/nn/activation_function.h"
#include "vi/la/matrix.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

float random(float start_range, float end_range) {
  float value(((float)rand()) / ((float)RAND_MAX));
  float range(end_range - start_range);
  return start_range + value * range;
}
}

namespace vi {
namespace nn {

layer::layer() : _activation(nullptr), _weights() {}

layer::layer(vi::la::context& context, std::shared_ptr<activation_function> activation,
             size_t output_count, size_t input_count)
    : _activation(activation), _weights(context, output_count, input_count + 1, 1.0f) {
  // initialize weights randomly to break symmetry
  float epsilon = sqrt(6.0f) / sqrt(input_count + output_count);
  for (size_t m = 0U; m < _weights.row_count(); ++m) {
    for (size_t n = 0U; n < _weights.column_count(); ++n) {
      _weights[m][n] = random(-epsilon, epsilon);
    }
  }
}

layer::layer(std::shared_ptr<activation_function> activation, const vi::la::matrix& weights)
    : _activation(activation), _weights(weights) {}

layer::layer(const layer& other) : _activation(other._activation), _weights(other._weights) {}

layer& layer::operator=(const layer& other) {
  if (this == &other) {
    return *this;
  }

  _activation = other.activation();
  _weights = other.weights();

  return *this;
}

vi::la::matrix layer::forward(const vi::la::matrix& input) const {
  const vi::la::matrix bias(context(), input.row_count(), 1U, 1.0);
  vi::la::matrix z((bias << input) * weights().transpose());
  _activation->activate(z);
  return z;
}

std::pair<vi::la::matrix, vi::la::matrix> layer::backward(const vi::la::matrix& inputs,
                                                          const vi::la::matrix& activations,
                                                          const vi::la::matrix& error) const {
  const vi::la::matrix derivative(_activation->gradient(activations));
  const vi::la::matrix delta = derivative.elementwise_product(error);

  const vi::la::matrix input_bias(activations.owning_context(), inputs.row_count(), 1U, 1.0);
  const vi::la::matrix biased_inputs((input_bias << inputs));
  const vi::la::matrix gradient = (delta.transpose() * biased_inputs) / -1.0;

  const vi::la::matrix delta_out = delta * weights();
  return std::make_pair(delta_out.columns(1U, delta_out.column_count() - 1U), gradient);
}

size_t layer::input_count() const {
  // bias unit is internal to the layer
  return weights().column_count() - 1;
}

size_t layer::output_count() const { return weights().row_count(); }

std::shared_ptr<activation_function> layer::activation() const { return _activation; }

void layer::activation(std::shared_ptr<activation_function> activation) {
  _activation = activation;
}

const vi::la::matrix& layer::weights() const { return _weights; }

void layer::weights(const vi::la::matrix& weights) { _weights = weights; }

vi::la::context& layer::context() { return _weights.owning_context(); }

vi::la::context& layer::context() const { return _weights.owning_context(); }
}
}
