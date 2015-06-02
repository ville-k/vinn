#ifndef __vinn__regularizer__
#define __vinn__regularizer__

#include <vi/la/matrix.h>

namespace vi {
namespace nn {

class l2_regularizer {
public:
  l2_regularizer(double weight_decay);

  std::pair<double, vi::la::matrix> penalty(const vi::la::matrix& weights) const;

private:
  double _weight_decay;
};
}
}

#endif /* defined(__vinn__regularizer__) */
