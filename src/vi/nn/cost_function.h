#ifndef __vinn__cost_function__
#define __vinn__cost_function__

#include <vi/la/matrix.h>

namespace vi {
namespace nn {

class cost_function {
public:
  virtual ~cost_function() {}

  virtual vi::la::matrix cost(const vi::la::matrix& expected, const vi::la::matrix& actual) = 0;
  virtual vi::la::matrix cost_derivative(const vi::la::matrix& expected,
                                         const vi::la::matrix& actual) = 0;
};

class cross_entropy_cost : public cost_function {
public:
  vi::la::matrix cost(const vi::la::matrix& expected, const vi::la::matrix& actual);
  vi::la::matrix cost_derivative(const vi::la::matrix& expected, const vi::la::matrix& actual);
};

class squared_error_cost : public cost_function {
public:
  vi::la::matrix cost(const vi::la::matrix& expected, const vi::la::matrix& actual);
  vi::la::matrix cost_derivative(const vi::la::matrix& expected, const vi::la::matrix& actual);
};
}
}

#endif
