#ifndef __vinn__opencl_matrix__
#define __vinn__opencl_matrix__

#include <vi/la/opencl/opencl_context.h>
#include <vi/la/matrix_implementation.h>

namespace cl {
class Buffer;
}

namespace vi {
namespace la {
namespace opencl {

/// Matrix implementation for OpenCL context
class matrix : public vi::la::matrix_implementation {
public:
  matrix(opencl_context& context, size_t rows, size_t columns, const float* initial_values);
  virtual ~matrix();

  virtual size_t row_count() const;
  virtual size_t column_count() const;

  virtual vi::la::context& owning_context() const;

  virtual float* raw_data();

  cl::Buffer* get();

private:
  size_t value_count() const;
  opencl_context& _context;
  size_t _row_count;
  size_t _column_count;
  cl::Buffer* _device_buffer;
  float* _host_buffer;
};
}
}
}

#endif
