#ifndef __vinn__layer__
#define __vinn__layer__

#include <vi/la/matrix.h>
#include <vi/la/context.h>
#include "activation_function.h"

namespace vi {
namespace nn {

class activation_function;

class layer {
public:
  layer();
  layer(vi::la::context& context, std::shared_ptr<activation_function> activation,
        size_t output_count, size_t input_count);
  layer(std::shared_ptr<activation_function> activation, const vi::la::matrix& weights);
  layer(const layer& other);

  layer& operator=(const layer& other);

  vi::la::matrix forward(const vi::la::matrix& input) const;

  std::pair<vi::la::matrix, vi::la::matrix> backward(const vi::la::matrix& input,
                                                     const vi::la::matrix& activations,
                                                     const vi::la::matrix& error) const;

  size_t input_count() const;
  size_t output_count() const;

  std::shared_ptr<activation_function> activation() const;
  void activation(std::shared_ptr<activation_function> activation);

  const vi::la::matrix& weights() const;
  void weights(const vi::la::matrix& weights);

  vi::la::context& context();
  vi::la::context& context() const;

private:
  std::shared_ptr<activation_function> _activation;
  vi::la::matrix _weights;
};
}
}

#endif
