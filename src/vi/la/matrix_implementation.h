#ifndef __vinn__matrix_implementation__
#define __vinn__matrix_implementation__

#include <cstddef>

namespace vi {
namespace la {

class context;

/// Interface that context specific matrices must conform to
class matrix_implementation {
public:
  virtual size_t row_count() const = 0;
  virtual size_t column_count() const = 0;

  virtual vi::la::context& owning_context() const = 0;

  virtual float* raw_data() = 0;
};
}
}

#endif
