#include <cstring>
#include "matrix_cl__source.h"
namespace vi {
namespace la {
namespace opencl_generated {
void matrix_cl__source(const char** name, const char** data, size_t& length) {
  *name = "matrix.cl";
  *data =
      "#if defined(DOUBLE_SUPPORT_AVAILABLE)\n#pragma OPENCL EXTENSION cl_khr_fp64 : "
      "enable\ntypedef double real_t;\n#else\ntypedef float real_t;\n#endif\n\n__kernel void "
      "matrix_multiply(__global real_t * product, __global real_t * operand_1, __global real_t * "
      "operand_2,\n                              size_t m, size_t n, size_t z) {\n  // operand_1: "
      "m x n\n  // operand_2: n x z\n  // product:   m x z\n  size_t row    = get_global_id(0);\n  "
      "size_t column = get_global_id(1);\n  size_t matrix_column = column;\n  size_t matrix_row = "
      "row;\n\n  real_t inner_product = 0.0;\n  for (size_t l = 0; l < n; ++l) {\n    "
      "inner_product += operand_1[(matrix_row * n + l)] * operand_2[(matrix_column + l * z)];\n  "
      "}\n  product[(matrix_row * z + matrix_column)] = inner_product;\n}\n\n__kernel void "
      "matrix_scalar_multiply(__global real_t * product, __global real_t * operand_1, real_t "
      "operand_2,\n                                   size_t m, size_t n) {\n  size_t row = "
      "get_global_id(0);\n  size_t col = get_global_id(1);\n\n  real_t value = operand_2 * "
      "operand_1[row * n + col];\n  product[row * n + col] = value;\n}\n\n__kernel void "
      "matrix_elementwise_multiply(__global real_t * product, __global real_t * operand_1, "
      "__global real_t * operand_2,\n                                        size_t m, size_t n) "
      "{\n  size_t row = get_global_id(0);\n  size_t col = get_global_id(1);\n\n  real_t value = "
      "operand_1[row * n + col] * operand_2[row * n + col];\n  product[row * n + col] = "
      "value;\n}\n\n__kernel void scalar_add(__global real_t * sum, __global real_t * operand_1, "
      "real_t operand_2,\n                       size_t m, size_t n) {\n  size_t row = "
      "get_global_id(0);\n  size_t col = get_global_id(1);\n\n  real_t value = operand_1[row * n + "
      "col] + operand_2;\n  sum[row * n + col] = value;\n}\n\n__kernel void matrix_add(__global "
      "real_t * sum, __global real_t * operand_1, __global real_t * operand_2,\n                   "
      "         size_t m, size_t n) {\n  size_t row = get_global_id(0);\n  size_t col = "
      "get_global_id(1);\n\n  real_t value = operand_1[row * n + col] + operand_2[row * n + "
      "col];\n  sum[row * n + col] = value;\n}\n\n__kernel void matrix_subtract(__global real_t * "
      "difference, __global real_t * operand_1, __global real_t * operand_2,\n                     "
      "                   size_t m, size_t n) {\n  size_t row = get_global_id(0);\n  size_t col = "
      "get_global_id(1);\n\n  real_t value = operand_1[row * n + col] - operand_2[row * n + "
      "col];\n  difference[row * n + col] = value;\n}\n\n__kernel void matrix_merge(__global "
      "real_t * merged, __global real_t * operand_1, __global real_t * operand_2,\n                "
      "         size_t rows, size_t operand_1_columns, size_t operand_2_columns) {\n  size_t row   "
      " = get_global_id(0U);\n\n  size_t merged_columns = operand_1_columns + "
      "operand_2_columns;\n\n  for (size_t i = 0U; i < operand_1_columns; ++i) {\n    merged[row * "
      "merged_columns + i] = operand_1[row * operand_1_columns + i];\n  }\n\n  for (size_t i = 0U; "
      "i < operand_2_columns; ++i) {\n    merged[row * merged_columns + operand_1_columns + i] = "
      "operand_2[row * operand_2_columns + i];\n  }\n}\n\n__kernel void matrix_transpose(__global "
      "real_t * transposed, __global real_t * original,\n                             size_t "
      "original_rows, size_t original_columns) {\n  size_t row    = get_global_id(0U);\n  size_t "
      "column = get_global_id(1U);\n\n  transposed[column * original_rows + row] = original[row * "
      "original_columns + column];\n}\n\n__kernel void sum_rows(__global real_t * summed, __global "
      "real_t * original, size_t rows, size_t columns) {\n  size_t col = get_global_id(1);\n\n  "
      "real_t sum = 0.0;\n  for (size_t row = 0U; row < rows; ++row) {\n    sum += original[row * "
      "columns + col];\n  }\n  summed[col] = sum;\n}\n\n__kernel void sum_columns(__global real_t "
      "* summed, __global real_t * original, size_t rows, size_t columns) {\n  size_t row = "
      "get_global_id(0);\n\n  for (size_t col = 0U; col < columns; ++col) {\n    summed[row] += "
      "original[row * columns + col];\n  }\n}\n\n__kernel void matrix_log(__global real_t * "
      "logged, __global real_t * original, size_t rows, size_t columns) {\n  size_t row = "
      "get_global_id(0);\n  size_t col = get_global_id(1);\n\n  real_t value = original[row * "
      "columns + col];\n  logged[row * columns + col] = log(value);\n}\n\n";
  length = std::strlen(*data) + 1U;
  return;
}
}
}
}
