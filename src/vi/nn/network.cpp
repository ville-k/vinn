#include "vi/nn/network.h"
#include "vi/nn/cost_function.h"
#include "vi/la/context.h"

#include <algorithm>
#include <sstream>
#include <string>

namespace vi {
namespace nn {

la::matrix network::forward(const la::matrix& inputs) const {
  la::matrix activation(inputs);
  for (const std::shared_ptr<layer>& l : layers_) {
    activation = l->forward(activation);
  }
  return activation;
}

std::pair<float, std::vector<la::matrix>> network::backward(const la::matrix& features,
                                                            const la::matrix& targets,
                                                            cost_function& cost_function) {
  std::vector<vi::la::matrix> activations;
  activations.push_back(features);
  for (const std::shared_ptr<layer>& l : layers_) {
    const vi::la::matrix activation = l->forward(activations.back());
    activations.push_back(activation);
  }

  const vi::la::matrix& hypotheses(activations.back());
  const vi::la::matrix costs(cost_function.cost(targets, hypotheses));
  const float cost = features.owning_context().sum_rows(costs)[0][0];

  vi::la::matrix errors(cost_function.cost_derivative(targets, hypotheses));

  std::vector<vi::la::matrix> gradients;
  for_each(layers_.rbegin(), layers_.rend(), [&](const std::shared_ptr<layer>& l) {
    const vi::la::matrix& layer_inputs = *(activations.end() - 2);
    const vi::la::matrix& layer_activations = activations.back();
    std::pair<vi::la::matrix, vi::la::matrix> error_and_gradient =
        l->backward(layer_inputs, layer_activations, errors);
    activations.pop_back();

    errors = error_and_gradient.first;
    la::matrix& gradient = error_and_gradient.second;
    gradients.insert(gradients.begin(), gradient);
  });

  return make_pair(cost, gradients);
}

void network::add(std::shared_ptr<layer> new_layer) throw(invalid_configuration) {
  if (layers_.size() > 0) {
    if (layers_.back()->output_count() != new_layer->input_count()) {
      std::ostringstream details;
      details << "Layer at back has different number of outputs(" << layers_.back()->output_count()
              << ") than added layer has inputs (" << new_layer->input_count() << ").";
      throw invalid_configuration(details.str());
    }
  }

  layers_.push_back(new_layer);
}

size_t network::size() const { return layers_.size(); }

network::iterator network::begin() { return layers_.begin(); }

network::const_iterator network::begin() const { return layers_.begin(); }

network::iterator network::end() { return layers_.end(); }

network::const_iterator network::end() const { return layers_.end(); }
}
}
