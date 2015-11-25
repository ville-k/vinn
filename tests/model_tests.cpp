#include "test.h"
#include <iostream>

#include "vi/io/model.h"
#include "vi/la/cpu/cpu_context.h"
#include "vi/nn/activation_function.h"

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

TEST(model, round_trip) {
  const auto temp_dir_path = boost::filesystem::temp_directory_path();
  fs::path model_path(temp_dir_path);
  model_path /= "round_trip_test.model";

  vi::la::cpu_context context;
  vi::nn::network out_network;
  out_network.add(std::make_shared<vi::nn::layer>(
      context, std::make_shared<vi::nn::linear_activation>(), 5, 5));
  out_network.add(std::make_shared<vi::nn::layer>(
      context, std::make_shared<vi::nn::hyperbolic_tangent>(), 5, 5));
  out_network.add(std::make_shared<vi::nn::layer>(
      context, std::make_shared<vi::nn::sigmoid_activation>(), 4, 5));
  out_network.add(std::make_shared<vi::nn::layer>(
      context, std::make_shared<vi::nn::softmax_activation>(), 2, 4));
  vi::io::model model(model_path.string());
  model.store(out_network);

  vi::io::model in_model(model_path.string());
  vi::nn::network in_network;
  in_model.load(in_network, context);

  EXPECT_EQ(out_network.size(), in_network.size());

  vi::nn::network::const_iterator out_iterator = out_network.begin();
  vi::nn::network::const_iterator in_iterator = in_network.begin();
  for (; out_iterator != out_network.end(); ++out_iterator, ++in_iterator) {
    std::shared_ptr<vi::nn::layer> out_layer = *out_iterator;
    std::shared_ptr<vi::nn::layer> in_layer = *in_iterator;

    EXPECT_MATRIX_EQ(out_layer->weights(), in_layer->weights());
    EXPECT_EQ(typeid(out_layer->activation()), typeid(in_layer->activation()));
  }

  fs::remove_all(model_path);
}
