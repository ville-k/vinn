#if defined(DOUBLE_SUPPORT_AVAILABLE)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real_t;
#else
typedef float real_t;
#endif

__kernel void matrix_multiply(__global real_t * product, __global real_t * operand_1, __global real_t * operand_2,
                              size_t m, size_t n, size_t z) {
  // operand_1: m x n
  // operand_2: n x z
  // product:   m x z
  size_t row    = get_global_id(0);
  size_t column = get_global_id(1);
  size_t matrix_column = column;
  size_t matrix_row = row;

  real_t inner_product = 0.0;
  for (size_t l = 0; l < n; ++l) {
    inner_product += operand_1[(matrix_row * n + l)] * operand_2[(matrix_column + l * z)];
  }
  product[(matrix_row * z + matrix_column)] = inner_product;
}

__kernel void matrix_scalar_multiply(__global real_t * product, __global real_t * operand_1, real_t operand_2,
                                   size_t m, size_t n) {
  size_t row = get_global_id(0);
  size_t col = get_global_id(1);

  real_t value = operand_2 * operand_1[row * n + col];
  product[row * n + col] = value;
}

__kernel void matrix_elementwise_multiply(__global real_t * product, __global real_t * operand_1, __global real_t * operand_2,
                                        size_t m, size_t n) {
  size_t row = get_global_id(0);
  size_t col = get_global_id(1);

  real_t value = operand_1[row * n + col] * operand_2[row * n + col];
  product[row * n + col] = value;
}

__kernel void scalar_add(__global real_t * sum, __global real_t * operand_1, real_t operand_2,
                       size_t m, size_t n) {
  size_t row = get_global_id(0);
  size_t col = get_global_id(1);

  real_t value = operand_1[row * n + col] + operand_2;
  sum[row * n + col] = value;
}

__kernel void matrix_add(__global real_t * sum, __global real_t * operand_1, __global real_t * operand_2,
                            size_t m, size_t n) {
  size_t row = get_global_id(0);
  size_t col = get_global_id(1);

  real_t value = operand_1[row * n + col] + operand_2[row * n + col];
  sum[row * n + col] = value;
}

__kernel void matrix_subtract(__global real_t * difference, __global real_t * operand_1, __global real_t * operand_2,
                                        size_t m, size_t n) {
  size_t row = get_global_id(0);
  size_t col = get_global_id(1);

  real_t value = operand_1[row * n + col] - operand_2[row * n + col];
  difference[row * n + col] = value;
}

__kernel void matrix_merge(__global real_t * merged, __global real_t * operand_1, __global real_t * operand_2,
                         size_t rows, size_t operand_1_columns, size_t operand_2_columns) {
  size_t row    = get_global_id(0U);

  size_t merged_columns = operand_1_columns + operand_2_columns;

  for (size_t i = 0U; i < operand_1_columns; ++i) {
    merged[row * merged_columns + i] = operand_1[row * operand_1_columns + i];
  }

  for (size_t i = 0U; i < operand_2_columns; ++i) {
    merged[row * merged_columns + operand_1_columns + i] = operand_2[row * operand_2_columns + i];
  }
}

__kernel void matrix_transpose(__global real_t * transposed, __global real_t * original,
                             size_t original_rows, size_t original_columns) {
  size_t row    = get_global_id(0U);
  size_t column = get_global_id(1U);

  transposed[column * original_rows + row] = original[row * original_columns + column];
}

__kernel void sum_rows(__global real_t * summed, __global real_t * original, size_t rows, size_t columns) {
  size_t col = get_global_id(1);

  real_t sum = 0.0;
  for (size_t row = 0U; row < rows; ++row) {
    sum += original[row * columns + col];
  }
  summed[col] = sum;
}

__kernel void sum_columns(__global real_t * summed, __global real_t * original, size_t rows, size_t columns) {
  size_t row = get_global_id(0);

  for (size_t col = 0U; col < columns; ++col) {
    summed[row] += original[row * columns + col];
  }
}

__kernel void matrix_log(__global real_t * logged, __global real_t * original, size_t rows, size_t columns) {
  size_t row = get_global_id(0);
  size_t col = get_global_id(1);

  real_t value = original[row * columns + col];
  logged[row * columns + col] = log(value);
}

