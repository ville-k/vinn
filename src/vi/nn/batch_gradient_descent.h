#ifndef __vinn__batch_gradient_descent__
#define __vinn__batch_gradient_descent__

#include <vi/nn/trainer.h>
#include <vi/nn/network.h>

namespace vi {
namespace nn {

class batch_gradient_descent : public trainer {
public:
  batch_gradient_descent(size_t max_epoch_count, double learning_rate);

  virtual double train(vi::nn::network& network, const vi::la::matrix& features,
                       const vi::la::matrix& targets,
                       vi::nn::cost_function& cost_function);

private:
  size_t _max_epoch_count;
  double _learning_rate;
};

}
}

#endif

