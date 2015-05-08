#if defined(DOUBLE_SUPPORT_AVAILABLE)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real_t;
#else
typedef float real_t;
#endif

__kernel void matrix_sigmoid(__global real_t * operand, size_t m, size_t n) {
  size_t row    = get_global_id(0);
  size_t column = get_global_id(1);

  real_t value = -1.0 * operand[row * n + column];
  operand[row * n + column] = 1.0 / (1.0 + exp(value));
}

__kernel void matrix_sigmoid_gradient(__global real_t * gradient, __global real_t * operand, size_t m, size_t n) {
  size_t row    = get_global_id(0);
  size_t column = get_global_id(1);

  real_t value = operand[row * n + column];
  gradient[row * n + column] = value * (1.0 - value);
}

__kernel void matrix_hyperbolic_tangent(__global real_t * operand, size_t m, size_t n) {
  size_t row    = get_global_id(0);
  size_t column = get_global_id(1);

  real_t value = operand[row * n + column];
  operand[row * n + column] = tanh(value);
}

__kernel void matrix_hyperbolic_tangent_gradient(__global real_t * gradient, __global real_t * operand, size_t m, size_t n) {
  size_t row    = get_global_id(0);
  size_t column = get_global_id(1);

  real_t value = operand[row * n + column];
  gradient[row * n + column] = 1.0 - (value * value);
}

__kernel void matrix_softmax_exp(__global real_t * operand, size_t m, size_t n) {
  size_t row    = get_global_id(0U);
  size_t column = get_global_id(1U);

  real_t value = exp(operand[row * n + column]);
  operand[row * n + column] = value;
}

__kernel void matrix_softmax_normalize(__global real_t * operand, size_t m, size_t n) {
  size_t row    = get_global_id(0U);

  real_t total = 0.0;
  for (size_t i = 0U; i < n; ++i) {
      total += operand[row * n + i];
  }

  for (size_t i = 0U; i < n; ++i) {
      operand[row * n + i] /= total;
  }
}

