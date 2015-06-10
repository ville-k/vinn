#include "benchmarks.h"
#include "vi/la.h"
#include <cmath>

static void BM_matrix_scalar_multiply(benchmark::State& state) {
  size_t context_index = state.range_x();
  size_t size = state.range_y();
  vi::la::context& context = *benchmarks::all_contexts()[context_index];

  vi::la::matrix m(context, size, size, 2.0);
  while (state.KeepRunning()) {
    vi::la::matrix result = m * 2.0;
  }

  size_t flops_per_iteration = size * size;
  size_t bytes_per_iteration = flops_per_iteration * sizeof(double);
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
  size_t bytes_per_iteration = flops_per_iteration * sizeof(double);
  state.SetBytesProcessed(state.iterations() * bytes_per_iteration);
  state.SetItemsProcessed(state.iterations() * flops_per_iteration);
}

static void all_contexts_10_to_1000(benchmark::internal::Benchmark* benchmark) {
  for (size_t context_index = 0U; context_index < benchmarks::all_contexts().size();
       ++context_index) {
    for (size_t exponent = 4; exponent < 10; ++exponent) {
      benchmark = benchmark->ArgPair(context_index, std::pow(2, exponent));
    }
  }
}

BENCHMARK(BM_matrix_scalar_multiply)->Apply(all_contexts_10_to_1000);
BENCHMARK(BM_matrix_matrix_multiply)->Apply(all_contexts_10_to_1000);
