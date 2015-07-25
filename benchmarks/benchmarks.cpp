#include "benchmarks.h"
#include "vi/la.h"

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
}

BENCHMARK_MAIN()
