#include "benchmarks.h"
#include "vi/nn.h"

static void BM_layer_forward(benchmark::State& state) {
  size_t context_index = state.range_x();
  size_t size = state.range_y();
  vi::la::context& context = *benchmarks::all_contexts()[context_index];

  vi::la::matrix weights(context, size, size, 0.5);
  vi::nn::layer layer(std::make_shared<vi::nn::sigmoid_activation>(), weights);
  const size_t example_count = 50;
  const size_t input_count = size - 1;
  vi::la::matrix inputs(context, example_count, input_count);
  while (state.KeepRunning()) {
    vi::la::matrix outputs = layer.forward(inputs);
  }

  size_t examples_per_iteration = example_count;
  size_t bytes_per_iteration = examples_per_iteration * input_count * sizeof(float);
  state.SetBytesProcessed(state.iterations() * bytes_per_iteration);
  state.SetItemsProcessed(state.iterations() * examples_per_iteration);
}

static void BM_layer_backward(benchmark::State& state) {
  size_t context_index = state.range_x();
  size_t size = state.range_y();
  vi::la::context& context = *benchmarks::all_contexts()[context_index];

  vi::la::matrix weights(context, size, size, 0.5);
  vi::nn::layer layer(std::make_shared<vi::nn::sigmoid_activation>(), weights);
  const size_t example_count = 50;
  const size_t input_count = size - 1;
  vi::la::matrix inputs(context, example_count, input_count, 0.5);
  vi::la::matrix activations(context, example_count, size, 0.8);
  vi::la::matrix delta(context, example_count, size, 0.2);

  while (state.KeepRunning()) {
    std::pair<vi::la::matrix, vi::la::matrix> delta_and_gradient =
        layer.backward(inputs, activations, delta);
  }

  size_t examples_per_iteration = example_count;
  size_t bytes_per_iteration = examples_per_iteration * input_count * sizeof(float);
  state.SetBytesProcessed(state.iterations() * bytes_per_iteration);
  state.SetItemsProcessed(state.iterations() * examples_per_iteration);
}

using benchmarks::all_contexts_16_to_512;

BENCHMARK(BM_layer_forward)->Apply(all_contexts_16_to_512);
BENCHMARK(BM_layer_backward)->Apply(all_contexts_16_to_512);
