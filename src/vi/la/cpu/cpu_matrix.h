#ifndef __vinn__cpu_matrix__
#define __vinn__cpu_matrix__

#include <vi/la/cpu/cpu_context.h>
#include <vi/la/matrix_implementation.h>

namespace vi {
namespace la {
namespace cpu {

/// Matrix implementation for CPU context
class matrix : public vi::la::matrix_implementation {
public:
  matrix(cpu_context& context, size_t rows, size_t columns, const float* initial_values);
  virtual ~matrix();

  size_t row_count() const;
  size_t column_count() const;

  virtual vi::la::context& owning_context() const;

  virtual float* raw_data();

  float* get();

private:
  cpu_context& _context;
  size_t _row_count;
  size_t _column_count;
  float* _buffer;
};
}
}
}

#endif
