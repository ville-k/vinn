#ifndef __vinn__layer__
#define __vinn__layer__

#include <vi/la/matrix.h>
#include <vi/la/context.h>

namespace vi {
namespace nn {

class activation_function;

class layer {
public:
  layer(vi::la::context& context, activation_function* activation, size_t output_count,
        size_t input_count);
  layer(vi::la::context& context, activation_function* activation, const vi::la::matrix& theta);
  layer(const layer& other);

  vi::la::matrix forward(const vi::la::matrix& input) const;

  std::pair<vi::la::matrix, vi::la::matrix> backward(const vi::la::matrix& input,
                                                     const vi::la::matrix& activations,
                                                     const vi::la::matrix& error) const;

  size_t get_input_count() const;
  size_t get_output_count() const;

  const vi::la::matrix& get_theta() const;
  void set_theta(const vi::la::matrix& theta);

private:
  activation_function* _activation;
  vi::la::matrix _theta;
  vi::la::context& _context;
};
}
}

#endif
