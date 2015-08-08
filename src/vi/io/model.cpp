#include "model.h"
#include "vi/io/csv_file.h"
#include "vi/io/network_deserializer.h"
#include "vi/io/network_serializer.h"
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace pt = boost::property_tree;
namespace fs = boost::filesystem;

namespace vi {
namespace io {
model::model(const std::string& path) : path_(path) {}

void model::load(vi::nn::network& network) {
  if (!fs::is_directory(model_dir_path())) {
    std::stringstream description;
    description << "Not a model directory: '" << model_dir_path() << "'";
    throw model::exception(description.str());
  }

  try {
    pt::ptree model_node;
    std::locale locale;
    pt::read_json(model_description_path().string(), model_node, locale);

    const pt::ptree network_node = model_node.get_child("network");
    network_deserializer deserializer(network);
    deserializer.deserialize(network_node);

    auto& layers = network.layers();
    for (size_t layer_index = 0U; layer_index < layers.size(); ++layer_index) {
      fs::fstream file_stream(layer_path(layer_index).string(), std::ios::in);

      vi::la::matrix weights(network.context(), 1, 1);
      vi::io::csv_file weight_file(file_stream);
      weight_file.load(weights);
      layers[layer_index].set_weights(weights);
    }
  } catch (pt::ptree_bad_path& e) {
    std::stringstream description;
    description << "Invalid model description: '" << e.what() << "'";
    throw model::exception(description.str());
  }
}

void model::store(const vi::nn::network& network) {
  if (fs::exists(model_dir_path())) {
    fs::remove_all(model_dir_path());
  }
  fs::create_directory(model_dir_path());

  boost::property_tree::ptree network_node;
  vi::io::network_serializer serializer(network);
  serializer.serialize(network_node);

  pt::ptree model_node;
  model_node.add_child("network", network_node);

  boost::property_tree::write_json(model_description_path().string(), model_node);

  fs::create_directory(model_data_dir_path());
  for (size_t layer_index = 0U; layer_index < network.layers().size(); ++layer_index) {
    fs::fstream file_stream(layer_path(layer_index), std::ios::out | std::ios::trunc);

    vi::io::csv_file weight_file(file_stream);
    weight_file.store(network.layers()[layer_index].get_weights());
  }
}

boost::filesystem::path model::model_dir_path() const { return boost::filesystem::path(path_); }

boost::filesystem::path model::model_description_path() const {
  return model_dir_path() /= "model.json";
}

boost::filesystem::path model::model_data_dir_path() const { return model_dir_path() /= "data"; }

boost::filesystem::path model::layer_path(size_t layer_index) const {
  std::stringstream file_name;
  file_name << layer_index << ".csv";
  return model_data_dir_path() /= file_name.str();
}
}
}
