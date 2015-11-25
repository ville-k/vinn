#ifndef __vinn__batch_gradient_descent__
#define __vinn__batch_gradient_descent__

#include <vi/nn/trainer.h>
#include <vi/nn/network.h>

namespace vi {
namespace nn {

class l2_regularizer;

class batch_gradient_descent : public trainer {
public:
  batch_gradient_descent(size_t max_epoch_count, float learning_rate);

  virtual float train(vi::nn::network& network, const vi::la::matrix& features,
                      const vi::la::matrix& targets, vi::nn::cost_function& cost_function);

  virtual float train(vi::nn::network& network, const vi::la::matrix& features,
                      const vi::la::matrix& targets, vi::nn::cost_function& cost_function,
                      const vi::nn::l2_regularizer& regularizer);

private:
  float train(vi::nn::network& network, const vi::la::matrix& features,
              const vi::la::matrix& targets, vi::nn::cost_function& cost_function,
              const vi::nn::l2_regularizer* regularizer);

  size_t _max_epoch_count;
  float _learning_rate;
};
}
}

#endif
