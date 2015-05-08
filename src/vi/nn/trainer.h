#ifndef __vinn__trainer__
#define __vinn__trainer__

#include <functional>

namespace vi {

namespace la {
class matrix;
}

namespace nn {

class network;
class cost_function;

class trainer {
public:
  virtual double train(vi::nn::network& network, const vi::la::matrix& features,
                       const vi::la::matrix& targets,
                       vi::nn::cost_function& cost_function) = 0;

  /// early stopping callback should return true to stop training, false to
  /// continue
  template <typename T> void set_stop_early(T stop_early) {
    _stop_early = stop_early;
  }

protected:
  std::function<bool(const vi::nn::network& network, size_t current_epoch,
                     double current_cost)> _stop_early;
};

}
}

#endif

