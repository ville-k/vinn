#include "layer_serializer.h"
#include "vi/nn/activation_function.h"
#include "vi/nn/layer.h"
#include <boost/property_tree/ptree.hpp>
#include <sstream>
#include <typeinfo>

namespace pt = boost::property_tree;

namespace vi {
namespace io {
layer_serializer::layer_serializer(const vi::nn::layer& layer) : layer_(layer) {}

void layer_serializer::serialize(boost::property_tree::ptree& layer_node) {
  const vi::nn::activation_function* a = layer_.activation().get();
  std::string activation_name("");
  if (typeid(*a) == typeid(vi::nn::sigmoid_activation)) {
    activation_name = "sigmoid";
  } else if (typeid(*a) == typeid(vi::nn::softmax_activation)) {
    activation_name = "softmax";
  } else if (typeid(*a) == typeid(vi::nn::hyperbolic_tangent)) {
    activation_name = "tanh";
  } else if (typeid(*a) == typeid(vi::nn::linear_activation)) {
    activation_name = "linear";
  } else {
    std::stringstream description;
    description << "invalid activation type: '" << typeid(*a).name() << "'";
    throw serializer::exception(description.str());
  }

  layer_node.put("activation_function", activation_name);
}
}
}
