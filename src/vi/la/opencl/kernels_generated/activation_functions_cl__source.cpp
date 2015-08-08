#include <cstring>
#include "activation_functions_cl__source.h"
namespace vi {
namespace la {
namespace opencl_generated {
void activation_functions_cl__source(const char** name, const char** data, size_t& length) {
  *name = "activation_functions.cl";
  *data = "#if defined(DOUBLE_SUPPORT_AVAILABLE)\n#pragma OPENCL EXTENSION cl_khr_fp64 : "
          "enable\ntypedef double real_t;\n#else\ntypedef float real_t;\n#endif\n\n__kernel void "
          "matrix_sigmoid(__global real_t * operand, size_t m, size_t n) {\n  size_t row    = "
          "get_global_id(0);\n  size_t column = get_global_id(1);\n\n  real_t value = -1.0 * "
          "operand[row * n + column];\n  operand[row * n + column] = 1.0 / (1.0 + "
          "exp(value));\n}\n\n__kernel void matrix_sigmoid_gradient(__global real_t * gradient, "
          "__global real_t * operand, size_t m, size_t n) {\n  size_t row    = get_global_id(0);\n "
          " size_t column = get_global_id(1);\n\n  real_t value = operand[row * n + column];\n  "
          "gradient[row * n + column] = value * (1.0 - value);\n}\n\n__kernel void "
          "matrix_hyperbolic_tangent(__global real_t * operand, size_t m, size_t n) {\n  size_t "
          "row    = get_global_id(0);\n  size_t column = get_global_id(1);\n\n  real_t value = "
          "operand[row * n + column];\n  operand[row * n + column] = tanh(value);\n}\n\n__kernel "
          "void matrix_hyperbolic_tangent_gradient(__global real_t * gradient, __global real_t * "
          "operand, size_t m, size_t n) {\n  size_t row    = get_global_id(0);\n  size_t column = "
          "get_global_id(1);\n\n  real_t value = operand[row * n + column];\n  gradient[row * n + "
          "column] = 1.0 - (value * value);\n}\n\n__kernel void matrix_softmax_exp(__global real_t "
          "* operand, size_t m, size_t n) {\n  size_t row    = get_global_id(0U);\n  size_t column "
          "= get_global_id(1U);\n\n  real_t value = exp(operand[row * n + column]);\n  operand[row "
          "* n + column] = value;\n}\n\n__kernel void matrix_softmax_normalize(__global real_t * "
          "operand, size_t m, size_t n) {\n  size_t row    = get_global_id(0U);\n\n  real_t total "
          "= 0.0;\n  for (size_t i = 0U; i < n; ++i) {\n      total += operand[row * n + i];\n  "
          "}\n\n  for (size_t i = 0U; i < n; ++i) {\n      operand[row * n + i] /= total;\n  "
          "}\n}\n\n";
  length = std::strlen(*data) + 1U;
  return;
}
}
}
}
