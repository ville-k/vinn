#ifndef __vinn__matrix__
#define __vinn__matrix__

#include <vi/la/context.h>
#include <vi/la/matrix_implementation.h>

#include <memory>
#include <stdexcept>
#include <vector>


namespace vi {
namespace la {

class matrix {
public:
  matrix();
  matrix(context& context,
         const std::initializer_list<std::initializer_list<double>>& rows);
  matrix(context& context, size_t rows, size_t columns,
         double initial_value = 0.0);
  matrix(context& context, const std::pair<size_t, size_t>& size,
         double initial_value = 0.0);
  matrix(context& context, size_t rows, size_t columns,
         const std::shared_ptr<double> values);

  matrix clone() const;

  matrix operator*(const matrix& other) const;
  matrix operator*(double const other) const;
  matrix elementwise_product(const matrix& other) const;
  matrix operator/(double const divisor) const;

  matrix operator+(const matrix& other) const;
  matrix operator+(const double other) const;

  matrix operator-(const matrix& other) const;
  matrix operator-(const double other) const;

  matrix operator<<(const matrix& other) const;

  double* operator[](size_t row_index) const;

  matrix columns(size_t start_column, size_t end_column) const;
  matrix column(size_t column_index) const;

  matrix rows(const std::vector<size_t>& row_indices) const;
  matrix rows(size_t start_row, size_t end_row) const;
  matrix row(size_t row_index) const;

  matrix sub_matrix(size_t start_row, size_t end_row, size_t start_column,
                    size_t end_column) const;

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

class incompatible_dimensions : public std::runtime_error {
public:
  incompatible_dimensions(const matrix& a, const matrix& b,
                          const std::string& operator_name);
  incompatible_dimensions(const std::string& error);

private:
  std::string incompatible_operands_message(const matrix& a, const matrix& b,
                                            const std::string& operator_name);
};

std::ostream& operator<<(std::ostream&, const vi::la::matrix&);

}
}

#endif
