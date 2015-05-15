#include "vi/nn/cost_function.h"
#include "vi/la/context.h"
#include <cmath>

namespace vi {
namespace nn {

vi::la::matrix cross_entropy_cost::cost(const vi::la::matrix& expected,
                                        const vi::la::matrix& actual) {

  vi::la::matrix logs(actual.owning_context(), actual.size());
  actual.owning_context().log(logs, actual);
  vi::la::matrix errors = expected.elementwise_product(logs);
  vi::la::matrix cost = expected.owning_context().sum_columns(errors) * -1.0;
  return cost;
}

vi::la::matrix cross_entropy_cost::cost_derivative(const vi::la::matrix& expected,
                                                   const vi::la::matrix& actual) {
  return actual - expected;
}

vi::la::matrix squared_error_cost::cost(const vi::la::matrix& expected,
                                        const vi::la::matrix& actual) {
  vi::la::matrix error = expected - actual;
  vi::la::matrix error_squared = error.elementwise_product(error);
  vi::la::matrix cost = expected.owning_context().sum_columns(error_squared) / 2.0;
  return cost;
}

vi::la::matrix squared_error_cost::cost_derivative(const vi::la::matrix& expected,
                                                   const vi::la::matrix& actual) {
  return expected - actual;
}
}
}
