#include "network_serializer.h"
#include "vi/io/layer_serializer.h"

namespace pt = boost::property_tree;

namespace vi {
namespace io {
network_serializer::network_serializer(const vi::nn::network& network) : network_(network) {}

void network_serializer::serialize(boost::property_tree::ptree& network_node) {
  pt::ptree layers_node;
  for (size_t layer_index = 0U; layer_index < network_.layers().size(); ++layer_index) {
    const auto& layer = network_.layers()[layer_index];
    pt::ptree layer_node;
    vi::io::layer_serializer layer_serializer(layer);
    layer_serializer.serialize(layer_node);

    std::stringstream layer_id;
    layer_id << layer_index;

    layers_node.add_child(layer_id.str(), layer_node);
  }
  network_node.add_child("layers", layers_node);
}
}
}
