#include "benchmarks.h"
#include "vi/la.h"
#include <cmath>

namespace benchmarks {

std::vector<vi::la::context*> all_contexts() {
  static std::vector<vi::la::context*> contexts = {};

  if (contexts.size() == 0) {
    contexts.push_back(new vi::la::cpu_context());

    std::vector<cl_device_id> device_ids = vi::la::opencl_context::supported_devices();
    for (cl_device_id device_id : device_ids) {
      contexts.push_back(new vi::la::opencl_context({device_id}));
    }
  }

  return contexts;
}

void all_contexts_16_to_512(benchmark::internal::Benchmark* benchmark) {
  for (size_t context_index = 0U; context_index < benchmarks::all_contexts().size();
       ++context_index) {
    for (size_t exponent = 4; exponent < 10; ++exponent) {
      benchmark = benchmark->ArgPair(context_index, std::pow(2, exponent));
    }
  }
}
}

BENCHMARK_MAIN();
