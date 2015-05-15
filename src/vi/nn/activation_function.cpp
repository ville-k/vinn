#include "vi/nn/activation_function.h"
#include "vi/la/context.h"

namespace vi {
namespace nn {

activation_function* sigmoid_activation::clone() const { return new sigmoid_activation(*this); }

void sigmoid_activation::activate(vi::la::matrix& inputs) const {
  inputs.owning_context().sigmoid(inputs);
}

vi::la::matrix sigmoid_activation::gradient(const vi::la::matrix& activations) const {
  vi::la::matrix gradient(activations.owning_context(), activations.size());
  activations.owning_context().sigmoid_gradient(gradient, activations);
  return gradient;
}

activation_function* softmax_activation::clone() const { return new softmax_activation(*this); }

void softmax_activation::activate(vi::la::matrix& inputs) const {
  inputs.owning_context().softmax(inputs);
}

vi::la::matrix softmax_activation::gradient(const vi::la::matrix& activations) const {
  vi::la::matrix gradient(activations.owning_context(), activations.size().first,
                          activations.size().second, -1.0);
  return gradient;
}

activation_function* hyperbolic_tangent::clone() const { return new hyperbolic_tangent(*this); }

void hyperbolic_tangent::activate(vi::la::matrix& inputs) const {
  inputs.owning_context().hyperbolic_tangent(inputs);
}

vi::la::matrix hyperbolic_tangent::gradient(const vi::la::matrix& activations) const {
  vi::la::matrix gradient(activations.owning_context(), activations.size());
  activations.owning_context().hyperbolic_tangent_gradient(gradient, activations);
  return gradient;
}

activation_function* linear_activation::clone() const { return new linear_activation(*this); }

void linear_activation::activate(vi::la::matrix& inputs) const {
  (void)inputs;
  return;
}

vi::la::matrix linear_activation::gradient(const vi::la::matrix& activations) const {
  vi::la::matrix gradient(activations.owning_context(), activations.size(), 1.0);
  return gradient;
}
}
}
