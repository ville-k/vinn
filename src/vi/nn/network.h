#ifndef __vinn__network__
#define __vinn__network__

#include <vi/la/context.h>
#include <vi/nn/layer.h>
#include <vi/la/matrix.h>

#include <vector>
#include <stdexcept>

namespace vi {
namespace nn {

class cost_function;

/// Neural network
class network {
public:
  class invalid_configuration : public std::runtime_error {
  public:
    invalid_configuration(const std::string& error) : std::runtime_error(error) {}
  };

  /// Construct network, throwing invalid_configuration if layer dimensions
  /// are incompatible
  network(vi::la::context& context, const std::vector<layer>& layers);

  /// Forward pass through network
  /// return predictions
  vi::la::matrix forward(const vi::la::matrix& features) const;

  /// Forward and backward pass through network
  /// return cost and gradients for each layer
  std::pair<double, std::vector<vi::la::matrix>> backward(const vi::la::matrix& features,
                                                          const vi::la::matrix& targets,
                                                          cost_function& cost_function);

  std::vector<layer>& layers();
  const std::vector<layer>& layers() const;

  vi::la::context& context();

private:
  vi::la::context& _context;
  std::vector<layer> _layers;
};
}
}

#endif
