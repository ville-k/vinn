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
  vi::nn::network out_network(context,
                              {vi::nn::layer(context, new vi::nn::linear_activation(), 5, 5),
                               vi::nn::layer(context, new vi::nn::hyperbolic_tangent(), 5, 5),
                               vi::nn::layer(context, new vi::nn::sigmoid_activation(), 4, 5),
                               vi::nn::layer(context, new vi::nn::softmax_activation(), 2, 4)});
  vi::io::model model(model_path.string());
  model.store(out_network);

  vi::io::model in_model(model_path.string());
  vi::nn::network in_network(context, {});
  in_model.load(in_network);

  EXPECT_EQ(out_network.layers().size(), in_network.layers().size());
  for (size_t i = 0U; i < out_network.layers().size(); ++i) {
    const auto& out_layer = out_network.layers()[i];
    const auto& in_layer = in_network.layers()[i];

    EXPECT_MATRIX_EQ(out_layer.get_weights(), in_layer.get_weights());
    EXPECT_EQ(typeid(out_layer.activation()), typeid(in_layer.activation()));
  }

  fs::remove_all(model_path);
}
