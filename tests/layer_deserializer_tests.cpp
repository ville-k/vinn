#include "test.h"
#include "vi/io/layer_deserializer.h"
#include "vi/nn/layer.h"
#include "vi/la/cpu/cpu_context.h"
#include "vi/nn/activation_function.h"
#include "../src/vi/nn/activation_function.h"
#include "../src/vi/io/serializer.h"
#include "../src/vi/io/deserializer.h"

namespace pt = boost::property_tree;

TEST(layer_deserializer, deserializes_sigmoid_layer) {
  pt::ptree layer_node;
  layer_node.put("activation_function", "sigmoid");

  vi::la::cpu_context context;
  vi::nn::layer layer(context, nullptr, 1, 1);
  vi::io::layer_deserializer deserializer(layer);
  deserializer.deserialize(layer_node);

  vi::nn::activation_function* activation = layer.activation().get();
  EXPECT_EQ(typeid(vi::nn::sigmoid_activation), typeid(*activation));
}

TEST(layer_deserializer, deserializes_softmax_layer) {
  pt::ptree layer_node;
  layer_node.put("activation_function", "softmax");

  vi::la::cpu_context context;
  vi::nn::layer layer(context, nullptr, 1, 1);
  vi::io::layer_deserializer deserializer(layer);
  deserializer.deserialize(layer_node);

  vi::nn::activation_function* activation = layer.activation().get();
  EXPECT_EQ(typeid(vi::nn::softmax_activation), typeid(*activation));
}

TEST(layer_deserializer, deserializes_tanh_layer) {
  pt::ptree layer_node;
  layer_node.put("activation_function", "tanh");

  vi::la::cpu_context context;
  vi::nn::layer layer(context, nullptr, 1, 1);
  vi::io::layer_deserializer deserializer(layer);
  deserializer.deserialize(layer_node);

  vi::nn::activation_function* activation = layer.activation().get();
  EXPECT_EQ(typeid(vi::nn::hyperbolic_tangent), typeid(*activation));
}

TEST(layer_deserializer, deserializes_linear_layer) {
  pt::ptree layer_node;
  layer_node.put("activation_function", "linear");

  vi::la::cpu_context context;
  vi::nn::layer layer(context, nullptr, 1, 1);
  vi::io::layer_deserializer deserializer(layer);
  deserializer.deserialize(layer_node);

  vi::nn::activation_function* activation = layer.activation().get();
  EXPECT_EQ(typeid(vi::nn::linear_activation), typeid(*activation));
}

TEST(layer_deserializer, fails_to_deserialize_unknown_activation_function) {
  pt::ptree layer_node;
  layer_node.put("activation_function", "hypergolic");

  vi::la::cpu_context context;
  vi::nn::layer layer(context, nullptr, 1, 1);
  vi::io::layer_deserializer deserializer(layer);
  EXPECT_THROW(deserializer.deserialize(layer_node), vi::io::deserializer::exception);
}
