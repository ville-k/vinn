#ifndef __vinn__network__
#define __vinn__network__

#include <vi/la/context.h>
#include <vi/nn/layer.h>
#include <vi/la/matrix.h>

#include <list>
#include <stdexcept>

namespace vi {
namespace nn {

class cost_function;

///
class invalid_configuration : public std::runtime_error {
public:
  invalid_configuration(const std::string& error) : std::runtime_error(error) {}
};

/// A feedforward neural network consists of a sequence of layers and knows
/// how to forward and backward propagate input and output data
class network {
public:
  /// Forward pass the provided input features trough the network
  /// \param features matrix with each row containing an
  ///        input vector
  /// return predictions with each row containing an output vector for
  ///        a corresponding row in the inputs
  vi::la::matrix forward(const vi::la::matrix& features) const;

  /// Forward and backward pass through the network
  /// \param features matrix with each row containing an
  ///        input vector
  /// \param targets matrix with each row containing a
  ///        target output vector
  /// \param cost_function cost function to use to evaluate
  ///        how much input features deviate from the output
  ///        targets
  /// return cost and gradients for each layer
  std::pair<float, std::vector<vi::la::matrix>> backward(const vi::la::matrix& features,
                                                         const vi::la::matrix& targets,
                                                         cost_function& cost_function);

  /// Push a layer on top of existing layers
  /// \param new_layer layer to be added
  /// \throw invalid_configuration if the number of layer inputs does not match
  ///        the number of outputs of the preceding layer
  void add(std::shared_ptr<layer> new_layer) throw(invalid_configuration);

  /// Number of layers contained in this network
  /// \return number of layers
  size_t size() const;

  /// Immutable iterator based access to contained layers
  typedef std::list<std::shared_ptr<layer>>::const_iterator const_iterator;
  /// Mutable iterator based access to contained layers
  typedef std::list<std::shared_ptr<layer>>::iterator iterator;

  /// \return iterator to first layer
  iterator begin();

  /// \return iterator to first layer
  const_iterator begin() const;

  /// \return iterator to the end of the contained layers
  iterator end();

  /// \return iterator to the end of the contained layers
  const_iterator end() const;

private:
  std::list<std::shared_ptr<layer>> layers_;
};
}
}

#endif
