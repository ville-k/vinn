#ifndef __vinn__context__
#define __vinn__context__

#include <memory>

namespace vi {
namespace la {

class matrix;
class matrix_implementation;

/// Interface that compute contexes must conform to
class context {
public:
  virtual ~context() {}

  virtual void multiply(matrix& product, const matrix& operand_1, const matrix& operand_2) = 0;
  virtual void multiply(matrix& product, const matrix& operand_1, const float operand_2) = 0;
  virtual void multiply_elementwise(matrix& product, const matrix& operand_1,
                                    const matrix& operand_2) = 0;

  virtual void add(matrix& sum, const matrix& operand_1, const float operand_2) = 0;
  virtual void add(matrix& sum, const matrix& operand_1, const matrix& operand_2) = 0;
  virtual void subtract(matrix& difference, const matrix& operand_1, const matrix& operand_2) = 0;

  virtual void sigmoid(matrix& operand) = 0;
  virtual void sigmoid_gradient(matrix& gradient, const matrix& operand) = 0;

  virtual void hyperbolic_tangent(matrix& operand) = 0;
  virtual void hyperbolic_tangent_gradient(matrix& gradient, const matrix& operand) = 0;

  virtual void softmax(matrix& operand) = 0;

  virtual void merge(matrix& merged, const matrix& operand_1, const matrix& operand_2) = 0;
  virtual void transpose(matrix& transposed, const matrix& original) = 0;

  virtual matrix sum_rows(const matrix& matrix) = 0;
  virtual matrix sum_columns(const matrix& matrix) = 0;
  virtual void log(matrix& result, const matrix& original) = 0;

  virtual void sub_matrix(matrix& target, const matrix& source, size_t start_row, size_t end_row,
                          size_t start_column, size_t end_column) = 0;

  virtual void convolve_2d(matrix& result, const matrix& mask, const matrix& original,
                           size_t channels) = 0;

  virtual std::shared_ptr<vi::la::matrix_implementation>
  implement_matrix(size_t rows, size_t columns, const float* initial_values) = 0;
};
}
}

#endif
