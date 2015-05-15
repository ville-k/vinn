#include "vi/nn/layer.h"
#include "vi/nn/activation_function.h"
#include "vi/la/matrix.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

double random(double start_range, double end_range) {
  double value(((double)rand()) / ((double)RAND_MAX));
  double range(end_range - start_range);
  return start_range + value * range;
}
}

namespace vi {
namespace nn {

layer::layer(vi::la::context& context, activation_function* activation, size_t output_count,
             size_t input_count)
    : _activation(activation), _theta(context, output_count, input_count + 1, 1.0),
      _context(context) {
  // random initialization
  double epsilon = sqrt(6.0) / sqrt(input_count + output_count);
  for (size_t m = 0U; m < _theta.row_count(); ++m) {
    for (size_t n = 0U; n < _theta.column_count(); ++n) {
      _theta[m][n] = random(-epsilon, epsilon);
    }
  }
}

layer::layer(vi::la::context& context, activation_function* activation, const vi::la::matrix& theta)
    : _activation(activation), _theta(theta), _context(context) {}

layer::layer(const layer& other)
    : _activation(other._activation), _theta(other._theta), _context(other._context) {}

vi::la::matrix layer::forward(const vi::la::matrix& input) const {
  const vi::la::matrix bias(_context, input.row_count(), 1U, 1.0);
  vi::la::matrix z((bias << input) * _theta.transpose());
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
  const vi::la::matrix gradient =
      (delta.transpose() * biased_inputs) / (-1.0 * activations.row_count());

  const vi::la::matrix delta_out = delta * _theta;
  return std::make_pair(delta_out.columns(1U, delta_out.column_count() - 1U), gradient);
}

size_t layer::get_input_count() const {
  // bias unit is internal to the layer
  return _theta.column_count() - 1;
}

size_t layer::get_output_count() const { return _theta.row_count(); }

const vi::la::matrix& layer::get_theta() const { return _theta; }

void layer::set_theta(const vi::la::matrix& theta) { _theta = theta; }
}
}
