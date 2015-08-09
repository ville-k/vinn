#ifndef __vinn__running_average__
#define __vinn__running_average__

#include <cstddef>
#include <vector>

namespace vi {
namespace nn {

class running_average {
public:
  running_average(size_t max_values);
  void add_value(float value);
  float calculate() const;

private:
  size_t _max_values;
  std::vector<float> _values;
};
}
}

#endif
