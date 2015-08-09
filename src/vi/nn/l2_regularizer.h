#ifndef __vinn__regularizer__
#define __vinn__regularizer__

#include <vi/la/matrix.h>

namespace vi {
namespace nn {

class l2_regularizer {
public:
  l2_regularizer(float weight_decay);

  std::pair<float, vi::la::matrix> penalty(const vi::la::matrix& weights) const;

private:
  float _weight_decay;
};
}
}

#endif /* defined(__vinn__regularizer__) */
