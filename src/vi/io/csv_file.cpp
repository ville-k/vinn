#include "csv_file.h"

#include <sstream>
#include <string>
#include <vector>

namespace vi {
namespace io {

csv_file::csv_file(std::iostream& stream) : _stream(stream) {}

csv_file::~csv_file() {}

vi::la::matrix csv_file::load(vi::la::context& context) {
  _stream.exceptions(std::iostream::badbit);
  std::vector<std::vector<double>> matrix_values;
  size_t max_columns(0U);

  std::string line;
  while (std::getline(_stream, line, '\n')) {
    std::vector<double> row;

    std::istringstream row_stream(line);
    std::string value;
    while (std::getline(row_stream, value, ',')) {
      row.push_back(atof(value.c_str()));
    }
    max_columns = std::max(max_columns, row.size());
    matrix_values.push_back(row);
  }

  double* values = new double[matrix_values.size() * max_columns];
  size_t offset = 0U;
  for (auto row : matrix_values) {
    for (size_t column = 0U; column < row.size(); ++column) {
      values[offset + column] = row[column];
    }

    offset += max_columns;
  }
  // matrix copies values - ensure they get released
  std::shared_ptr<double> shared_values(values, [](double* p) { delete[] p; });

  return vi::la::matrix(context, matrix_values.size(), max_columns, shared_values);
}
}
}
