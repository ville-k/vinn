#ifndef __mlcl__opencl_context__
#define __mlcl__opencl_context__

#include <vi/la/context.h>
#include <memory>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace cl {
class Context;
class CommandQueue;
class Kernel;
}

namespace vi {
namespace la {

class opencl_context : public context {
public:
  opencl_context(const std::vector<cl_device_id>& device_ids);
  virtual ~opencl_context();

  static std::vector<cl_device_id>
  supported_devices(cl_device_type device_type = CL_DEVICE_TYPE_ALL);

  std::shared_ptr<vi::la::matrix_implementation> implement_matrix(size_t rows, size_t columns,
                                                                  const float* initial_values);

  void multiply(matrix& product, const matrix& operand_1, const matrix& operand_2);
  void multiply(matrix& product, const matrix& operand_1, const float operand_2);
  void multiply_elementwise(matrix& product, const matrix& operand_1, const matrix& operand_2);

  void add(matrix& sum, const matrix& operand_1, const float operand_2);
  void add(matrix& sum, const matrix& operand_1, const matrix& operand_2);
  void subtract(matrix& difference, const matrix& operand_1, const matrix& operand_2);

  void sigmoid(matrix& operand);
  void sigmoid_gradient(matrix& gradient, const matrix& operand);
  void hyperbolic_tangent(matrix& operand);
  void hyperbolic_tangent_gradient(matrix& gradient, const matrix& operand);
  void softmax(matrix& operand);

  void merge(matrix& merged, const matrix& operand_1, const matrix& operand_2);
  void transpose(matrix& transposed, const matrix& original);

  matrix sum_rows(const matrix& original);
  matrix sum_columns(const matrix& original);
  void log(matrix& result, const matrix& original);

  void sub_matrix(matrix& target, const matrix& original, size_t start_row, size_t end_row,
                  size_t start_column, size_t end_column);

  void convolve_2d(matrix& result, const matrix& mask, const matrix& original, size_t channels);

  cl::Context& context();
  cl::CommandQueue& command_queue();

private:
  void load_kernels();

  class private_members;

  cl::Context* _context;
  cl::CommandQueue* _command_queue;

  cl::Kernel* _matrix_multiply_kernel;
  cl::Kernel* _matrix_scalar_multiply;
  cl::Kernel* _matrix_elementwise_multiply;

  cl::Kernel* _matrix_add;
  cl::Kernel* _scalar_add;
  cl::Kernel* _matrix_subtract;

  cl::Kernel* _matrix_sigmoid_kernel;
  cl::Kernel* _sigmoid_gradient;
  cl::Kernel* _hyperbolic_tangent;
  cl::Kernel* _hyperbolic_tangent_gradient;
  cl::Kernel* _matrix_softmax_exp_kernel;
  cl::Kernel* _matrix_softmax_normalize_kernel;

  cl::Kernel* _matrix_merge_kernel;
  cl::Kernel* _matrix_transpose_kernel;

  cl::Kernel* _sum_rows;
  cl::Kernel* _sum_columns;
  cl::Kernel* _log;

  cl::Kernel* _convolve_2d;
};
}
}

#endif
