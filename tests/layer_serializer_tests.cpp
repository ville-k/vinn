#include "test.h"
#include "vi/io/layer_serializer.h"
#include "vi/nn/layer.h"
#include "vi/la/cpu/cpu_context.h"

namespace pt = boost::property_tree;

TEST(layer_serializer, serializes_sigmoid_layer) {
  vi::la::cpu_context context;
  vi::nn::sigmoid_activation sigmoid;
  vi::nn::layer layer(context, std::make_shared<vi::nn::sigmoid_activation>(), 1, 1);
  vi::io::layer_serializer serializer(layer);

  pt::ptree layer_node;
  serializer.serialize(layer_node);
  EXPECT_EQ("sigmoid", layer_node.get<std::string>("activation_function"));
}

TEST(layer_serializer, serializes_tanh_layer) {
  vi::la::cpu_context context;
  vi::nn::layer layer(context, std::make_shared<vi::nn::hyperbolic_tangent>(), 1, 1);
  vi::io::layer_serializer serializer(layer);

  pt::ptree layer_node;
  serializer.serialize(layer_node);
  EXPECT_EQ("tanh", layer_node.get<std::string>("activation_function"));
}

TEST(layer_serializer, serializes_softmax_layer) {
  vi::la::cpu_context context;
  vi::nn::layer layer(context, std::make_shared<vi::nn::softmax_activation>(), 1, 1);
  vi::io::layer_serializer serializer(layer);

  pt::ptree layer_node;
  serializer.serialize(layer_node);
  EXPECT_EQ("softmax", layer_node.get<std::string>("activation_function"));
}

TEST(layer_serializer, serializes_linear_layer) {
  vi::la::cpu_context context;
  vi::nn::layer layer(context, std::make_shared<vi::nn::linear_activation>(), 1, 1);
  vi::io::layer_serializer serializer(layer);

  pt::ptree layer_node;
  serializer.serialize(layer_node);
  EXPECT_EQ("linear", layer_node.get<std::string>("activation_function"));
}

namespace test {
class unsupported_activation : public vi::nn::activation_function {
public:
  activation_function* clone() const { return new unsupported_activation(); }

  void activate(vi::la::matrix& inputs) const { inputs.size(); }

  vi::la::matrix gradient(const vi::la::matrix& activations) const { return activations; }
};
}

TEST(layer_serializer, fails_serializing_unsupported_activation_function) {
  vi::la::cpu_context context;
  vi::nn::layer layer(context, std::make_shared<test::unsupported_activation>(), 1, 1);
  vi::io::layer_serializer serializer(layer);

  pt::ptree layer_node;
  EXPECT_THROW(serializer.serialize(layer_node), vi::io::serializer::exception);
}
