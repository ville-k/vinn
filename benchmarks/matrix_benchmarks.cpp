#include "benchmarks.h"
#include "vi/la.h"

static void BM_matrix_scalar_multiply(benchmark::State& state) {
  size_t context_index = state.range_x();
  size_t size = state.range_y();
  vi::la::context& context = *benchmarks::all_contexts()[context_index];

  vi::la::matrix m(context, size, size, 2.0);
  while (state.KeepRunning()) {
    vi::la::matrix result = m * 2.0;
  }

  size_t flops_per_iteration = size * size;
  size_t bytes_per_iteration = flops_per_iteration * sizeof(float);
  state.SetBytesProcessed(state.iterations() * bytes_per_iteration);
  state.SetItemsProcessed(state.iterations() * flops_per_iteration);
}

static void BM_matrix_matrix_multiply(benchmark::State& state) {
  size_t context_index = state.range_x();
  size_t size = state.range_y();
  vi::la::context& context = *benchmarks::all_contexts()[context_index];

  vi::la::matrix a(context, size, size, 2.0);
  vi::la::matrix b(context, size, size, 2.0);
  while (state.KeepRunning()) {
    vi::la::matrix result = a * b;
  }

  // x^2 * (x multiplies + x additions)
  size_t flops_per_iteration = (size + size) * size * size;
  size_t bytes_per_iteration = flops_per_iteration * sizeof(float);
  state.SetBytesProcessed(state.iterations() * bytes_per_iteration);
  state.SetItemsProcessed(state.iterations() * flops_per_iteration);
}

static void BM_matrix_sub_matrix(benchmark::State& state) {
  size_t context_index = state.range_x();
  size_t size = state.range_y();
  vi::la::context& context = *benchmarks::all_contexts()[context_index];

  const size_t MAX_SIZE = 512;

  vi::la::matrix m(context, MAX_SIZE, MAX_SIZE, 2.0);
  while (state.KeepRunning()) {
    vi::la::matrix sub_matrix = m.sub_matrix(0, size - 1, 0, size - 1);
  }

  size_t flops_per_iteration = size * size;
  size_t bytes_per_iteration = flops_per_iteration * sizeof(float);
  state.SetBytesProcessed(state.iterations() * bytes_per_iteration);
  state.SetItemsProcessed(state.iterations() * flops_per_iteration);
}

static void BM_matrix_transpose(benchmark::State& state) {
  size_t context_index = state.range_x();
  size_t size = state.range_y();
  vi::la::context& context = *benchmarks::all_contexts()[context_index];

  vi::la::matrix m(context, size, size, 2.0);
  while (state.KeepRunning()) {
    vi::la::matrix result = m.transpose();
  }

  size_t flops_per_iteration = size * size;
  size_t bytes_per_iteration = flops_per_iteration * sizeof(float);
  state.SetBytesProcessed(state.iterations() * bytes_per_iteration);
  state.SetItemsProcessed(state.iterations() * flops_per_iteration);
}

using benchmarks::all_contexts_16_to_512;

BENCHMARK(BM_matrix_scalar_multiply)->Apply(all_contexts_16_to_512);
BENCHMARK(BM_matrix_matrix_multiply)->Apply(all_contexts_16_to_512);
BENCHMARK(BM_matrix_sub_matrix)->Apply(all_contexts_16_to_512);
BENCHMARK(BM_matrix_transpose)->Apply(all_contexts_16_to_512);
