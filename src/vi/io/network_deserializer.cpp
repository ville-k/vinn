#include "vi/io/network_deserializer.h"
#include "vi/io/layer_deserializer.h"

namespace pt = boost::property_tree;

namespace vi {
namespace io {
network_deserializer::network_deserializer(vi::nn::network& network) : network_(network) {}

void network_deserializer::deserialize(const boost::property_tree::ptree& network_node,
                                       vi::la::context& context) {
  pt::ptree layers_node = network_node.get_child("layers");
  for (const auto& layer_node : layers_node) {
    std::shared_ptr<vi::nn::layer> layer = std::make_shared<vi::nn::layer>(context, nullptr, 1, 1);
    layer_deserializer deserializer(*layer.get());
    deserializer.deserialize(layer_node.second);
    network_.add(layer);
  }
}
}
}
