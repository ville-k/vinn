#include "test.h"

#include "vi/io/network_serializer.h"
#include "vi/la/cpu/cpu_context.h"

namespace pt = boost::property_tree;

TEST(network_serializer, serializes_layers) {
  vi::la::cpu_context context;
  const vi::nn::network network(context,
                                {vi::nn::layer(context, new vi::nn::sigmoid_activation(), 4, 5),
                                 vi::nn::layer(context, new vi::nn::sigmoid_activation(), 4, 4)});

  pt::ptree serialized;
  vi::io::network_serializer serializer(network);
  serializer.serialize(serialized);
  EXPECT_NO_THROW(serialized.get_child("layers"));
  EXPECT_EQ(2U, serialized.get_child("layers").size());
}
