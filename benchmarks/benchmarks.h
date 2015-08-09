#ifndef __vinn__benchmarks__
#define __vinn__benchmarks__

#include "benchmark/benchmark.h"

namespace vi {
namespace la {
class context;
}
}

namespace benchmarks {
std::vector<vi::la::context*> all_contexts();
void all_contexts_16_to_512(benchmark::internal::Benchmark* benchmark);
}

#endif
