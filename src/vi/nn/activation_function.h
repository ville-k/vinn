#ifndef __vinn__activation_function__
#define __vinn__activation_function__

#include <vi/la/matrix.h>

namespace vi {
namespace nn {

/// Interface that activation functions must conform to
class activation_function {
public:
  virtual ~activation_function() {}

  virtual activation_function* clone() const = 0;
  virtual void activate(vi::la::matrix& inputs) const = 0;
  virtual vi::la::matrix gradient(const vi::la::matrix& activations) const = 0;
};

class sigmoid_activation : public activation_function {
public:
  activation_function* clone() const;
  void activate(vi::la::matrix& inputs) const;
  vi::la::matrix gradient(const vi::la::matrix& activations) const;
};

class softmax_activation : public activation_function {
public:
  activation_function* clone() const;
  void activate(vi::la::matrix& inputs) const;
  vi::la::matrix gradient(const vi::la::matrix& activations) const;
};

class hyperbolic_tangent : public activation_function {
public:
  virtual activation_function* clone() const;
  void activate(vi::la::matrix& inputs) const;
  vi::la::matrix gradient(const vi::la::matrix& activations) const;
};

class linear_activation : public activation_function {
public:
  virtual activation_function* clone() const;
  void activate(vi::la::matrix& inputs) const;
  vi::la::matrix gradient(const vi::la::matrix& activations) const;
};
}
}

#endif
