#include "layer_deserializer.h"
#include <memory>

namespace vi {
namespace io {

layer_deserializer::layer_deserializer(vi::nn::layer& layer) : layer_(layer) {}

void layer_deserializer::deserialize(const boost::property_tree::ptree& layer_node) {
  const std::string activation_name = layer_node.get<std::string>("activation_function");
  std::shared_ptr<vi::nn::activation_function> activation;
  if (activation_name == "sigmoid") {
    activation.reset(new vi::nn::sigmoid_activation);
  } else if (activation_name == "softmax") {
    activation.reset(new vi::nn::softmax_activation);
  } else if (activation_name == "tanh") {
    activation.reset(new vi::nn::hyperbolic_tangent);
  } else if (activation_name == "linear") {
    activation.reset(new vi::nn::linear_activation);
  } else {
    std::stringstream description;
    description << "invalid activation type: '" << activation_name << "'";
    throw deserializer::exception(description.str());
  }
  layer_.activation(activation);
}
}
}
