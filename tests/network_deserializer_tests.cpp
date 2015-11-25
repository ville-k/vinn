#include "test.h"
#include "vi/io/network_deserializer.h"
#include "vi/la/cpu/cpu_context.h"

namespace pt = boost::property_tree;

TEST(network_deserializer, deserializes_layers) {
  vi::la::cpu_context context;
  vi::nn::network network;

  pt::ptree serialized;
  pt::ptree layers_node;
  pt::ptree layer_node;
  layer_node.put("activation_function", "sigmoid");
  layers_node.add_child("0", layer_node);
  serialized.add_child("layers", layers_node);

  vi::io::network_deserializer deserializer(network);
  deserializer.deserialize(serialized, context);
  EXPECT_EQ(1U, network.size());
}
