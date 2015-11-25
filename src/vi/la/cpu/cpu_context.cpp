#include "vi/la/cpu/cpu_context.h"
#include "vi/la/cpu/cpu_matrix.h"
#include "vi/la/matrix.h"

#include <cassert>
#include <cmath>
#include <memory>
#include <cfenv>
#include <iostream>

using std::cout;
using std::endl;

namespace vi {
namespace la {

std::shared_ptr<vi::la::matrix_implementation>
cpu_context::implement_matrix(size_t rows, size_t columns, const float* initial_values) {
  matrix_implementation* impl = new cpu::matrix(*this, rows, columns, initial_values);
  return std::shared_ptr<matrix_implementation>(impl);
}

void cpu_context::multiply(matrix& product, const matrix& operand_1, const matrix& operand_2) {
  cpu::matrix* product_impl = dynamic_cast<cpu::matrix*>(product.implementation());
  cpu::matrix* operand_1_impl = dynamic_cast<cpu::matrix*>(operand_1.implementation());
  cpu::matrix* operand_2_impl = dynamic_cast<cpu::matrix*>(operand_2.implementation());

  for (size_t m = 0U; m < product.row_count(); ++m) {
    for (size_t n = 0U; n < product.column_count(); ++n) {
      float value(0.0);
      for (size_t j = 0U; j < operand_1.column_count(); ++j) {
        value += operand_1_impl->get()[m * operand_1.column_count() + j] *
                 operand_2_impl->get()[j * operand_2.column_count() + n];
      }
      product_impl->get()[m * product.column_count() + n] = value;
    }
  }
}

void cpu_context::multiply(matrix& product, const matrix& operand_1, const float operand_2) {
  cpu::matrix* product_impl = dynamic_cast<cpu::matrix*>(product.implementation());
  cpu::matrix* operand_1_impl = dynamic_cast<cpu::matrix*>(operand_1.implementation());

  for (size_t m = 0U; m < product.row_count(); ++m) {
    for (size_t n = 0U; n < product.column_count(); ++n) {
      product_impl->get()[m * product.column_count() + n] =
          operand_1_impl->get()[m * product.column_count() + n] * operand_2;
    }
  }
}

void cpu_context::multiply_elementwise(matrix& product, const matrix& operand_1,
                                       const matrix& operand_2) {
  cpu::matrix* product_impl = dynamic_cast<cpu::matrix*>(product.implementation());
  cpu::matrix* operand_1_impl = dynamic_cast<cpu::matrix*>(operand_1.implementation());
  cpu::matrix* operand_2_impl = dynamic_cast<cpu::matrix*>(operand_2.implementation());

  for (size_t m = 0U; m < product.row_count(); ++m) {
    for (size_t n = 0U; n < product.column_count(); ++n) {
      const float value = operand_1_impl->get()[m * operand_1.column_count() + n] *
                          operand_2_impl->get()[m * operand_2.column_count() + n];
      product_impl->get()[m * product.column_count() + n] = value;
    }
  }
}

void cpu_context::add(matrix& sum, const matrix& operand_1, const float operand_2) {
  cpu::matrix* sum_impl = dynamic_cast<cpu::matrix*>(sum.implementation());
  cpu::matrix* operand_1_impl = dynamic_cast<cpu::matrix*>(operand_1.implementation());

  for (size_t m = 0U; m < sum.row_count(); ++m) {
    for (size_t n = 0U; n < sum.column_count(); ++n) {
      const float value = operand_1_impl->get()[m * operand_1.column_count() + n];
      sum_impl->get()[m * sum.column_count() + n] = value + operand_2;
    }
  }
}

void cpu_context::add(matrix& sum, const matrix& operand_1, const matrix& operand_2) {
  cpu::matrix* sum_impl = dynamic_cast<cpu::matrix*>(sum.implementation());
  cpu::matrix* operand_1_impl = dynamic_cast<cpu::matrix*>(operand_1.implementation());
  cpu::matrix* operand_2_impl = dynamic_cast<cpu::matrix*>(operand_2.implementation());

  for (size_t m = 0U; m < sum.row_count(); ++m) {
    for (size_t n = 0U; n < sum.column_count(); ++n) {
      const float value = operand_1_impl->get()[m * operand_1.column_count() + n] +
                          operand_2_impl->get()[m * operand_2.column_count() + n];
      sum_impl->get()[m * sum.column_count() + n] = value;
    }
  }
}

void cpu_context::subtract(matrix& difference, const matrix& operand_1, const matrix& operand_2) {
  cpu::matrix* difference_impl = dynamic_cast<cpu::matrix*>(difference.implementation());
  cpu::matrix* operand_1_impl = dynamic_cast<cpu::matrix*>(operand_1.implementation());
  cpu::matrix* operand_2_impl = dynamic_cast<cpu::matrix*>(operand_2.implementation());

  for (size_t m = 0U; m < difference.row_count(); ++m) {
    for (size_t n = 0U; n < difference.column_count(); ++n) {
      const float value = operand_1_impl->get()[m * operand_1.column_count() + n] -
                          operand_2_impl->get()[m * operand_2.column_count() + n];
      difference_impl->get()[m * difference.column_count() + n] = value;
    }
  }
}

void cpu_context::sigmoid(matrix& operand) {
  cpu::matrix* impl = dynamic_cast<cpu::matrix*>(operand.implementation());
  float* b = impl->raw_data();
  for (size_t m = 0U; m < operand.row_count(); ++m) {
    for (size_t n = 0U; n < operand.column_count(); ++n) {
      const float input = b[m * operand.column_count() + n];
      const float output = 1.0 / (1.0 + ::expf(-1.0 * input));
      b[m * operand.column_count() + n] = output;
    }
  }
}

void cpu_context::sigmoid_gradient(matrix& gradient, const matrix& operand) {
  cpu::matrix* gradient_impl = dynamic_cast<cpu::matrix*>(gradient.implementation());
  cpu::matrix* operand_impl = dynamic_cast<cpu::matrix*>(operand.implementation());

  for (size_t m = 0U; m < gradient.row_count(); ++m) {
    for (size_t n = 0U; n < gradient.column_count(); ++n) {
      const float input = operand_impl->get()[m * operand.column_count() + n];
      const float output = input * (1.0 - input);
      gradient_impl->get()[m * gradient.column_count() + n] = output;
    }
  }
}

void cpu_context::hyperbolic_tangent(matrix& operand) {
  cpu::matrix* operand_impl = dynamic_cast<cpu::matrix*>(operand.implementation());

  for (size_t m = 0U; m < operand.row_count(); ++m) {
    for (size_t n = 0U; n < operand.column_count(); ++n) {
      float value = operand_impl->get()[m * operand.column_count() + n];
      float output = tanh(value);
      operand_impl->get()[m * operand.column_count() + n] = output;
    }
  }
}

void cpu_context::hyperbolic_tangent_gradient(matrix& gradient, const matrix& operand) {
  cpu::matrix* gradient_impl = dynamic_cast<cpu::matrix*>(gradient.implementation());
  cpu::matrix* operand_impl = dynamic_cast<cpu::matrix*>(operand.implementation());

  for (size_t m = 0U; m < gradient.row_count(); ++m) {
    for (size_t n = 0U; n < gradient.column_count(); ++n) {
      float value = operand_impl->get()[m * operand.column_count() + n];
      gradient_impl->get()[m * operand.column_count() + n] = 1.0 - (value * value);
    }
  }
}

void cpu_context::softmax(matrix& operand) {
  cpu::matrix* impl = dynamic_cast<cpu::matrix*>(operand.implementation());
  float* buffer = impl->get();
  for (size_t m = 0U; m < operand.row_count(); ++m) {
    float row_total(0.0);
    for (size_t n = 0U; n < operand.column_count(); ++n) {
      float input_value = buffer[m * operand.column_count() + n];
      float value = std::exp(input_value);
      row_total += value;
      buffer[m * operand.column_count() + n] = value;
    }

    for (size_t n = 0U; n < operand.column_count(); ++n) {
      buffer[m * operand.column_count() + n] /= row_total;
    }
  }
}

void cpu_context::merge(matrix& merged, const matrix& operand_1, const matrix& operand_2) {
  cpu::matrix* merged_impl = dynamic_cast<cpu::matrix*>(merged.implementation());
  float* merged_buffer = merged_impl->get();
  cpu::matrix* operand_1_impl = dynamic_cast<cpu::matrix*>(operand_1.implementation());
  float* operand_1_buffer = operand_1_impl->get();
  cpu::matrix* operand_2_impl = dynamic_cast<cpu::matrix*>(operand_2.implementation());
  float* operand_2_buffer = operand_2_impl->get();

  for (size_t m = 0U; m < merged.row_count(); ++m) {
    for (size_t n = 0U; n < operand_1.column_count(); ++n) {
      merged_buffer[m * merged.column_count() + n] =
          operand_1_buffer[m * operand_1.column_count() + n];
    }

    for (size_t n = 0U; n < operand_2.column_count(); ++n) {
      merged_buffer[m * merged.column_count() + operand_1.column_count() + n] =
          operand_2_buffer[m * operand_2.column_count() + n];
    }
  }
}

void cpu_context::transpose(matrix& transposed, const matrix& original) {
  cpu::matrix* transposed_impl = dynamic_cast<cpu::matrix*>(transposed.implementation());
  float* transposed_buffer = transposed_impl->get();
  cpu::matrix* original_impl = dynamic_cast<cpu::matrix*>(original.implementation());
  float* original_buffer = original_impl->get();

  for (size_t m = 0U; m < original.row_count(); ++m) {
    for (size_t n = 0U; n < original.column_count(); ++n) {
      transposed_buffer[n * original.row_count() + m] =
          original_buffer[m * original.column_count() + n];
    }
  }
}

matrix cpu_context::sum_rows(const matrix& original) {
  vi::la::matrix sums(*this, 1U, original.column_count(), 0.0);
  cpu::matrix* original_impl = dynamic_cast<cpu::matrix*>(original.implementation());
  cpu::matrix* sums_impl = dynamic_cast<cpu::matrix*>(sums.implementation());

  for (size_t m = 0U; m < original.row_count(); ++m) {
    for (size_t n = 0U; n < original.column_count(); ++n) {
      sums_impl->get()[n] += original_impl->get()[m * original.column_count() + n];
    }
  }

  return sums;
}

matrix cpu_context::sum_columns(const matrix& original) {
  vi::la::matrix sums(*this, original.row_count(), 1U, 0.0);
  cpu::matrix* original_impl = dynamic_cast<cpu::matrix*>(original.implementation());
  cpu::matrix* sums_impl = dynamic_cast<cpu::matrix*>(sums.implementation());
  for (size_t m = 0U; m < original.row_count(); ++m) {
    for (size_t n = 0U; n < original.column_count(); ++n) {
      sums_impl->get()[m] += original_impl->get()[m * original.column_count() + n];
    }
  }

  return sums;
}

void cpu_context::log(matrix& result, const matrix& original) {
  cpu::matrix* result_impl = dynamic_cast<cpu::matrix*>(result.implementation());
  cpu::matrix* original_impl = dynamic_cast<cpu::matrix*>(original.implementation());

  for (size_t m = 0U; m < result.row_count(); ++m) {
    for (size_t n = 0U; n < result.column_count(); ++n) {
      float value = original_impl->get()[m * original.column_count() + n];
      result_impl->get()[m * result.column_count() + n] = logf(value);
    }
  }
}

void cpu_context::sub_matrix(matrix& target, const matrix& original, size_t start_row,
                             size_t end_row, size_t start_column, size_t end_column) {
  cpu::matrix* target_impl = dynamic_cast<cpu::matrix*>(target.implementation());
  cpu::matrix* original_impl = dynamic_cast<cpu::matrix*>(original.implementation());

  const size_t sub_rows(end_row - start_row + 1U);
  const size_t sub_columns(end_column - start_column + 1U);

  for (size_t m = 0; m < sub_rows; ++m) {
    for (size_t n = 0; n < sub_columns; ++n) {
      size_t source_offset = (m + start_row) * original.column_count() + start_column + n;
      target_impl->get()[m * target.column_count() + n] = original_impl->get()[source_offset];
    }
  }
}

void cpu_context::convolve_2d(matrix& result, const matrix& mask, const matrix& original,
                              size_t channels) {
  size_t mask_width = mask.column_count();
  size_t mask_height = mask.row_count();

  size_t mask_horizontal_radius = mask_width / 2;
  size_t mask_vertical_radius = mask_height / 2;

  size_t horizontal_steps = result.column_count() / channels;
  size_t vertical_steps = result.row_count();

  for (size_t m = 0U; m < vertical_steps; ++m) {
    for (size_t n = 0U; n < horizontal_steps; ++n) {
      for (size_t channel = 0U; channel < channels; ++channel) {
        float value = 0.0;
        for (size_t mask_row = 0U; mask_row < mask_height; ++mask_row) {
          for (size_t mask_column = 0U; mask_column < mask_width; ++mask_column) {

            float mask_element = mask[mask_row][mask_column];
            float source_element = 0.0;
            if ((m + mask_row) >= mask_vertical_radius &&
                (m + mask_row) < (vertical_steps + mask_vertical_radius) &&
                (n + mask_column) >= mask_horizontal_radius &&
                (n + mask_column) < (horizontal_steps + mask_horizontal_radius)) {

              size_t source_row = m - mask_vertical_radius + mask_row;
              size_t source_column = n - mask_horizontal_radius + mask_column;

              source_element = original[source_row][channels * source_column + channel];
            } else {
              // ghost elements are zero
              source_element = 0.0;
            }

            value += source_element * mask_element;
          }
        }

        result[m][channels * n + channel] = value;
      }
    }
  }
}
}
}
