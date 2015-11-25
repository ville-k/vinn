#ifndef __mlcl__cpu_context__
#define __mlcl__cpu_context__

#include <vi/la/context.h>

namespace vi {
namespace la {

/// CPU/C++ based reference implementation of linear algebra operations
class cpu_context : public context {
public:
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

  matrix sum_rows(const matrix& operand);
  matrix sum_columns(const matrix& matrix);
  void log(matrix& result, const matrix& original);

  void sub_matrix(matrix& target, const matrix& original, size_t start_row, size_t end_row,
                  size_t start_column, size_t end_column);

  void convolve_2d(matrix& result, const matrix& mask, const matrix& original, size_t channels);
};
}
}

#endif
