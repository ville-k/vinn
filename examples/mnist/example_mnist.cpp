#include <vi/io.h>
#include <vi/la.h>
#include <vi/nn.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;
using namespace vi::io;
using namespace vi::la;
using namespace vi::nn;

namespace {

vector<size_t> generate_random_indices(size_t index_count) {
  vector<size_t> randomized_indices(index_count);
  for (size_t i = 0U; i < index_count; ++i) {
    randomized_indices.push_back(i);
  }
  random_shuffle(randomized_indices.begin(), randomized_indices.end());
  return randomized_indices;
}

pair<vi::la::matrix, vi::la::matrix>
load_svm_dataset(vi::la::context& context, const string& path,
                 size_t feature_count) {
  fstream stream(path);
  libsvm_file file(stream);
  return file.load_labels_and_features(context, feature_count);
}

}

int main(int, const char**) {
  auto devices = vi::la::opencl_context::supported_devices();
  if (devices.size() == 0) {
    cerr << "No supported OpenCL devices available" << endl;
    exit(1);
  }
  std::vector<cl_device_id> selected_devices = { devices[0] };
    
  // Load data
  const string training_set_path(string(SRCROOT) + "/examples/mnist/mnist.scale");
  const string test_set_path(string(SRCROOT) + "/examples/mnist/mnist.scale.t");
  const size_t feature_count = 780U;
  const double training_fraction = 0.9;

  vi::la::opencl_context context(selected_devices);

  cout << "Loading training dataset from: " << training_set_path << endl;
  const auto labels_and_features =
      load_svm_dataset(context, training_set_path, feature_count);
  const matrix all_labels(labels_and_features.first);
  const matrix all_features(labels_and_features.second);

  cout << "Loading testing dataset from: " << test_set_path << endl;
  auto test_labels_and_features =
      load_svm_dataset(context, test_set_path, feature_count);
  const matrix testing_features(test_labels_and_features.second);
  const matrix testing_labels(test_labels_and_features.first);

  // randomize examples and divide up to training and validation sets
  const vector<size_t> all_indices =
      generate_random_indices(all_features.row_count());
  const size_t training_example_count =
      training_fraction * all_indices.size();
  const vector<size_t> training_indices(
      all_indices.begin(), all_indices.begin() + training_example_count);
  const vector<size_t> validation_indices(
      all_indices.begin() + training_example_count, all_indices.end());

  const matrix training_features(all_features.rows(training_indices));
  const matrix training_labels(all_labels.rows(training_indices));

  const matrix validation_features(all_features.rows(validation_indices));
  const matrix validation_labels(all_labels.rows(validation_indices));

  cout << "training examples:   " << training_features.row_count() << endl;
  cout << "validation examples: " << validation_features.row_count() << endl;
  cout << "training input features: " << all_features.column_count() << endl;
  cout << "test examples:       " << testing_features.row_count() << endl;
  cout << "test input features: " << testing_features.column_count() << endl;

  // Build network
  layer l1(context, new hyperbolic_tangent(), 25, feature_count);
  layer l2(context, new softmax_activation(), 10, 25);
  network net(context, {l1, l2});

  // Train
  minibatch_gradient_descent minibatch(10U, 0.009, 50U);
  label_map map({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  minibatch.set_stop_early([&](const network& nw, size_t epoch, double cost)
                               -> bool {
    // stop early based on performance on the validation set
    matrix predictions = nw.forward(validation_features);

    result_measurements measurements(context, map.labels());
    measurements.add_results(validation_labels,
                             map.activations_to_labels(predictions));
    cout << "epoch: " << epoch << ", cost: " << cost
         << ", accuracy: " << measurements.accuracy() << endl;

    return measurements.accuracy() >= 0.95;
  });

  matrix target_activations = map.labels_to_activations(training_labels);
  cross_entropy_cost cost_function;
  minibatch.train(net, training_features, target_activations, cost_function);

  // Test
  matrix predictions = net.forward(testing_features);
  result_measurements measurements(context, map.labels());
  measurements.add_results(testing_labels,
                           map.activations_to_labels(predictions));
  cout << "Results on test set:" << endl;
  cout << measurements << endl;

  return 0;
}
