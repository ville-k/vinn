#include "test.h"

#include "vi/io/network_serializer.h"
#include "vi/la/cpu/cpu_context.h"

namespace pt = boost::property_tree;

TEST(network_serializer, serializes_layers) {
  vi::la::cpu_context context;
  vi::nn::network network;

  network.add(std::make_shared<vi::nn::layer>(
      context, std::make_shared<vi::nn::sigmoid_activation>(), 4, 5));
  network.add(std::make_shared<vi::nn::layer>(
      context, std::make_shared<vi::nn::sigmoid_activation>(), 4, 4));

  pt::ptree serialized;
  vi::io::network_serializer serializer(network);
  serializer.serialize(serialized);
  EXPECT_NO_THROW(serialized.get_child("layers"));
  EXPECT_EQ(2U, serialized.get_child("layers").size());
}
