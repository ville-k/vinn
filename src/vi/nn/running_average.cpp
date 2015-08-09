#include "vi/nn/running_average.h"

namespace vi {
namespace nn {

running_average::running_average(size_t max_values) : _max_values(max_values) {
  _values.reserve(max_values);
}

void running_average::add_value(float value) {
  if (_values.size() >= _max_values) {
    _values.pop_back();
  }
  _values.insert(_values.begin(), value);
}

float running_average::calculate() const {
  if (_values.size() == 0) {
    return 0.0;
  }

  float total(0.0);
  for (const float& value : _values) {
    total += value;
  }
  return total / _values.size();
}
}
}
