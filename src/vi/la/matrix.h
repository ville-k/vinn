#ifndef __vinn__matrix__
#define __vinn__matrix__

#include <vi/la/context.h>
#include <vi/la/matrix_implementation.h>

#include <memory>
#include <stdexcept>
#include <vector>

namespace vi {
namespace la {

class incompatible_dimensions : public std::runtime_error {
public:
  incompatible_dimensions(const matrix& a, const matrix& b, const std::string& operator_name);
  incompatible_dimensions(const std::string& error);

private:
  std::string incompatible_operands_message(const matrix& a, const matrix& b,
                                            const std::string& operator_name);
};

class matrix {
public:
  matrix();
  matrix(context& context, const std::initializer_list<std::initializer_list<float>>& rows) throw(
      incompatible_dimensions);
  matrix(context& context, size_t rows, size_t columns,
         float initial_value = 0.0) throw(incompatible_dimensions);
  matrix(context& context, const std::pair<size_t, size_t>& size,
         float initial_value = 0.0) throw(incompatible_dimensions);
  matrix(context& context, size_t rows, size_t columns,
         const std::shared_ptr<float> values) throw(incompatible_dimensions);
  matrix(context& context, const float* values, size_t rows,
         size_t columns) throw(incompatible_dimensions);

  matrix clone() const;

  matrix operator*(const matrix& other) const throw(incompatible_dimensions);
  matrix operator*(const float other) const;
  matrix elementwise_product(const matrix& other) const throw(incompatible_dimensions);
  matrix operator/(const float divisor) const;

  matrix operator+(const matrix& other) const throw(incompatible_dimensions);
  matrix operator+(const float other) const;

  matrix operator-(const matrix& other) const throw(incompatible_dimensions);
  matrix operator-(const float other) const;

  matrix operator<<(const matrix& other) const throw(incompatible_dimensions);
  float* operator[](size_t row_index) const throw(std::out_of_range);

  matrix columns(size_t start_column, size_t end_column) const throw(std::out_of_range);
  matrix column(size_t column_index) const throw(std::out_of_range);

  matrix rows(const std::vector<size_t>& row_indices) const throw(std::out_of_range);
  matrix rows(size_t start_row, size_t end_row) const throw(std::out_of_range);
  matrix row(size_t row_index) const throw(std::out_of_range);

  matrix sub_matrix(size_t start_row, size_t end_row, size_t start_column, size_t end_column) const
      throw(std::out_of_range);

  matrix transpose() const;

  size_t row_count() const;
  size_t column_count() const;
  std::pair<size_t, size_t> size() const;

  vi::la::context& owning_context() const;

private:
  friend class cpu_context;
  friend class opencl_context;
  friend std::ostream& operator<<(std::ostream&, const vi::la::matrix&);

  matrix(std::shared_ptr<vi::la::matrix_implementation> implementation);
  matrix_implementation* implementation() const;

  std::shared_ptr<vi::la::matrix_implementation> _implementation;
};

std::ostream& operator<<(std::ostream&, const vi::la::matrix&);
}
}

#endif
