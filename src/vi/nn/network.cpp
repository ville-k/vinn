#include "vi/nn/network.h"
#include "vi/nn/cost_function.h"

#include <algorithm>
#include <sstream>
#include <string>

namespace vi {
namespace nn {

network::network(vi::la::context& context, const std::vector<layer>& layers)
    : _context(context), _layers(layers) {
  // validate layer dimensions
  for (size_t i = 1U; i < _layers.size(); ++i) {
    if (_layers[i - 1].get_output_count() != _layers[i].get_input_count()) {
      std::ostringstream details;
      details << "Layer " << i << " has different number of inputs("
              << _layers[i].get_input_count() << ") than layer " << i - 1
              << " has outputs (" << _layers[i - 1].get_output_count() << ").";
      throw invalid_configuration(details.str());
    }
  }
}

la::matrix network::forward(const la::matrix& inputs) const {
  la::matrix activation(inputs);
  for (const layer& l : _layers) {
    activation = l.forward(activation);
  }
  return activation;
}

std::pair<double, std::vector<la::matrix>>
network::backward(const la::matrix& features, const la::matrix& targets,
                  cost_function& cost_function) {
  std::vector<vi::la::matrix> activations;
  activations.push_back(features);
  for (const layer& l : _layers) {
    const vi::la::matrix activation = l.forward(activations.back());
    activations.push_back(activation);
  }

  const vi::la::matrix& hypotheses(activations.back());
  const vi::la::matrix costs(cost_function.cost(targets, hypotheses));
  const double cost = _context.sum_rows(costs)[0][0] / costs.row_count();

  vi::la::matrix errors(cost_function.cost_derivative(targets, hypotheses));

  std::vector<vi::la::matrix> gradients;
  for_each(_layers.rbegin(), _layers.rend(), [&](const layer& l) {
    const vi::la::matrix& layer_inputs = *(activations.end() - 2);
    const vi::la::matrix& layer_activations = activations.back();
    std::pair<vi::la::matrix, vi::la::matrix> error_and_gradient =
        l.backward(layer_inputs, layer_activations, errors);
    activations.pop_back();

    errors = error_and_gradient.first;
    gradients.insert(gradients.begin(), error_and_gradient.second);
  });

  return make_pair(cost, gradients);
}

std::vector<layer>& network::layers() { return _layers; }
}

}

