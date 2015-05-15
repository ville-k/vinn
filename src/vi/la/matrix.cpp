#include "matrix.h"
#include "context.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

namespace vi {
namespace la {

matrix::matrix() {}

matrix::matrix(std::shared_ptr<vi::la::matrix_implementation> implementation)
    : _implementation(implementation) {}

matrix::matrix(vi::la::context& context,
               const std::initializer_list<std::initializer_list<double>>& rows) {
  size_t max_column_count(0U);
  for (const std::initializer_list<double>& row : rows) {
    max_column_count = std::max(max_column_count, row.size());
  }
  if (rows.size() == 0U || max_column_count == 0U) {
    throw incompatible_dimensions("Matrix cannot have have 0 rows or columns");
  }

  double* values = new double[rows.size() * max_column_count];
  size_t row_offset(0U);
  for (const std::initializer_list<double>& row : rows) {
    size_t colum_offset(0U);
    for (const double& element : row) {
      values[row_offset + colum_offset] = element;
      ++colum_offset;
    }
    row_offset += max_column_count;
  }
  _implementation = context.implement_matrix(rows.size(), max_column_count, values);
  delete[] values;
}

matrix::matrix(vi::la::context& context, size_t rows, size_t columns, double initial_value) {
  if (rows == 0U || columns == 0U) {
    throw incompatible_dimensions("Matrix cannot have have 0 rows or columns");
  }
  const size_t value_count(rows * columns);
  double* values = new double[value_count];
  for (size_t i = 0U; i < value_count; ++i) {
    values[i] = initial_value;
  }
  _implementation = context.implement_matrix(rows, columns, values);
  delete[] values;
}

matrix::matrix(vi::la::context& context, const std::pair<size_t, size_t>& size,
               double initial_value)
    : matrix(context, size.first, size.second, initial_value) {}

matrix::matrix(vi::la::context& context, size_t rows, size_t columns,
               const std::shared_ptr<double> values) {
  if (rows == 0U || columns == 0U) {
    throw incompatible_dimensions("Matrix cannot have have 0 rows or columns");
  }
  _implementation = context.implement_matrix(rows, columns, values.get());
}

matrix matrix::clone() const { return sub_matrix(0u, row_count() - 1U, 0U, column_count() - 1U); }

matrix matrix::operator*(matrix const& other) const {
  if (column_count() != other.row_count()) {
    throw incompatible_dimensions(*this, other, "*");
  }

  matrix product(owning_context(), row_count(), other.column_count());
  _implementation->owning_context().multiply(product, *this, other);
  return product;
}

matrix matrix::operator*(double const other) const {
  matrix product(owning_context(), row_count(), column_count());
  owning_context().multiply(product, *this, other);
  return product;
}

matrix matrix::elementwise_product(const matrix& other) const {
  if (row_count() != other.row_count() || column_count() != other.column_count()) {
    throw incompatible_dimensions(*this, other, ".*");
  }
  matrix product(owning_context(), row_count(), column_count());
  owning_context().multiply_elementwise(product, *this, other);
  return product;
}

matrix matrix::operator/(double const divisor) const { return *this * (1.0 / divisor); }

matrix matrix::operator+(const matrix& other) const {
  if (row_count() != other.row_count() || column_count() != other.column_count()) {
    throw vi::la::incompatible_dimensions(*this, other, "+");
  }

  vi::la::matrix sum(other.owning_context(), row_count(), column_count());
  owning_context().add(sum, *this, other);
  return sum;
}

matrix matrix::operator+(const double other) const {
  vi::la::matrix sum(owning_context(), row_count(), column_count());
  owning_context().add(sum, *this, other);
  return sum;
}

matrix matrix::operator-(const matrix& other) const {
  if (row_count() != other.row_count() || column_count() != other.column_count()) {
    throw vi::la::incompatible_dimensions(*this, other, "-");
  }

  vi::la::matrix difference(other.owning_context(), row_count(), column_count());
  owning_context().subtract(difference, *this, other);
  return difference;
}

matrix matrix::operator-(const double other) const {
  vi::la::matrix difference(owning_context(), row_count(), column_count());
  owning_context().add(difference, *this, -1.0 * other);
  return difference;
}

matrix matrix::operator<<(const matrix& other) const {
  if (row_count() != other.row_count()) {
    throw incompatible_dimensions("Combined matrices must have equal number of rows");
  }

  matrix merged(owning_context(), row_count(), column_count() + other.column_count(), 0.0);
  owning_context().merge(merged, *this, other);
  return merged;
}

double* matrix::operator[](size_t row_index) const {
  if (row_index >= row_count()) {
    std::ostringstream details;
    details << "Row index: " << row_index << " out of range:[0," << row_count() - 1 << "]";
    throw incompatible_dimensions(details.str());
  }
  double* buffer = _implementation->raw_data();
  return &buffer[row_index * column_count()];
}

matrix matrix::columns(size_t start_column, size_t end_column) const {
  return sub_matrix(0, row_count() - 1, start_column, end_column);
}

matrix matrix::column(size_t column_index) const { return columns(column_index, column_index); }

matrix matrix::rows(const std::vector<size_t>& row_indices) const {
  matrix selected(owning_context(), row_indices.size(), column_count());
  size_t m = 0U;
  for (const size_t& row_index : row_indices) {
    for (size_t n = 0U; n < column_count(); ++n) {
      selected[m][n] = (*this)[row_index][n];
    }
    ++m;
  }
  return selected;
}

matrix matrix::rows(size_t start_row, size_t end_row) const {
  return sub_matrix(start_row, end_row, 0U, column_count() - 1);
}

matrix matrix::row(size_t row_index) const { return rows(row_index, row_index); }

matrix matrix::sub_matrix(size_t start_row, size_t end_row, size_t start_column,
                          size_t end_column) const {
  if (start_row >= row_count() || end_row >= row_count() || start_row > end_row ||
      start_column >= column_count() || end_column >= column_count() || start_column > end_column) {
    std::ostringstream details;
    details << "Incompatible row or column range for matrix: ";
    details << row_count() << "x" << column_count();
    details << ", range: [" << start_row << ":" << end_row << "]";
    throw incompatible_dimensions(details.str());
  }

  const size_t sub_rows(end_row - start_row + 1U);
  const size_t sub_columns(end_column - start_column + 1U);

  matrix sub_matrix(owning_context(), sub_rows, sub_columns);
  owning_context().sub_matrix(sub_matrix, *this, start_row, end_row, start_column, end_column);
  return sub_matrix;
}

matrix matrix::transpose() const {
  matrix transposed(owning_context(), column_count(), row_count(), 0.0);
  owning_context().transpose(transposed, *this);
  return transposed;
}

size_t matrix::row_count() const { return _implementation->row_count(); }

size_t matrix::column_count() const { return _implementation->column_count(); }

std::pair<size_t, size_t> matrix::size() const {
  return std::make_pair(row_count(), column_count());
}

vi::la::context& matrix::owning_context() const { return _implementation->owning_context(); }

vi::la::matrix_implementation* matrix::implementation() const { return _implementation.get(); }

incompatible_dimensions::incompatible_dimensions(const matrix& a, const matrix& b,
                                                 const std::string& operator_name)
    : std::runtime_error(incompatible_operands_message(a, b, operator_name)) {}

incompatible_dimensions::incompatible_dimensions(const std::string& error)
    : std::runtime_error(error) {}

std::string
incompatible_dimensions::incompatible_operands_message(const matrix& a, const matrix& b,
                                                       const std::string& operator_name) {
  std::ostringstream details;
  details << "Incompatible dimensions: ";
  details << a.size().first << "x" << a.size().second;
  details << " " << operator_name << " ";
  details << b.size().first << "x" << b.size().second;
  return details.str();
}

std::ostream& operator<<(std::ostream& os, const vi::la::matrix& matrix) {
  for (size_t m = 0U; m < matrix.row_count(); ++m) {
    for (size_t n = 0U; n < matrix.column_count(); ++n) {
      os << matrix[m][n] << "\t";
    }
    os << "\n";
  }
  return os;
}
}
}
