#include "vi/la/opencl/opencl_matrix.h"
#include <cassert>
#include <CL/cl.hpp>

namespace vi {
namespace la {
namespace opencl {

matrix::matrix(opencl_context& context, size_t rows, size_t columns, const float* initial_values)
    : _context(context), _row_count(rows), _column_count(columns), _host_buffer(nullptr) {
  size_t value_count = rows * columns;
  if (initial_values) {
    _device_buffer = new cl::Buffer(context.context(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE,
                                    value_count * sizeof(cl_float), (void*)initial_values);
  } else {
    _device_buffer = new cl::Buffer(context.context(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
                                    value_count * sizeof(cl_float), nullptr);
  }

  _host_buffer = static_cast<float*>(context.command_queue().enqueueMapBuffer(
      *_device_buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0U, value_count * sizeof(cl_float)));
}

matrix::~matrix() {
  _context.command_queue().enqueueUnmapMemObject(*_device_buffer, static_cast<void*>(_host_buffer));
  delete _device_buffer;
}

size_t matrix::row_count() const { return _row_count; }

size_t matrix::column_count() const { return _column_count; }

vi::la::context& matrix::owning_context() const { return _context; }

float* matrix::raw_data() { return _host_buffer; }

cl::Buffer* matrix::get() { return _device_buffer; }

size_t matrix::value_count() const { return _row_count * _column_count; }
}
}
}
