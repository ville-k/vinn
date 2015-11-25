#include "csv_file.h"

#include <sstream>
#include <string>
#include <vector>

namespace vi {
namespace io {

csv_file::csv_file(std::iostream& stream, char delimiter)
    : _delimiter(delimiter), _stream(stream) {}

csv_file::~csv_file() {}

void csv_file::load(vi::la::matrix& matrix) { load(matrix, nullptr); }

void csv_file::load(vi::la::matrix& matrix, std::vector<std::string>& header) {
  load(matrix, &header);
}

void csv_file::load(vi::la::matrix& matrix, std::vector<std::string>* header) {
  _stream.exceptions(std::iostream::badbit);
  std::vector<std::vector<float>> matrix_values;
  size_t max_columns(0U);

  bool first_line(true);
  std::string line;
  while (std::getline(_stream, line, '\n')) {
    if (first_line && header) {
      first_line = false;
      parse_header(line, *header);
      continue;
    }

    std::vector<float> row;
    parse_row(line, row);
    max_columns = std::max(max_columns, row.size());
    matrix_values.push_back(row);
  }

  matrix = vi::la::matrix(matrix.owning_context(), matrix_values.size(), max_columns,
                          make_buffer(matrix_values, max_columns));
}

void csv_file::parse_header(const std::string& line, std::vector<std::string>& header) const {
  std::istringstream row_stream(line);
  std::string value;
  while (std::getline(row_stream, value, _delimiter)) {
    header.push_back(value);
  }
}

void csv_file::parse_row(const std::string& line, std::vector<float>& row) const {
  std::istringstream row_stream(line);
  std::string value;
  while (std::getline(row_stream, value, _delimiter)) {
    row.push_back(std::stof(value));
  }
}

std::shared_ptr<float> csv_file::make_buffer(const std::vector<std::vector<float>>& matrix_values,
                                             size_t max_columns) const {
  float* values = new float[matrix_values.size() * max_columns];
  size_t offset = 0U;
  for (auto row : matrix_values) {
    for (size_t column = 0U; column < row.size(); ++column) {
      values[offset + column] = row[column];
    }

    offset += max_columns;
  }

  return std::shared_ptr<float>(values, [](float* p) { delete[] p; });
}

void csv_file::store(const vi::la::matrix& matrix) { store(matrix, nullptr); }

void csv_file::store(const vi::la::matrix& matrix, std::vector<std::string>& header) {
  store(matrix, &header);
}

void csv_file::store(const vi::la::matrix& matrix, std::vector<std::string>* header) {
  _stream.precision(17);
  if (header) {
    for (size_t i = 0U; i < header->size(); ++i) {
      const std::string& column = header->at(i);
      _stream << column;

      if (i != header->size() - 1U) {
        _stream << _delimiter;
      }
    }
    _stream << "\n";
  }

  for (size_t row = 0U; row < matrix.row_count(); ++row) {
    for (size_t column = 0U; column < matrix.column_count(); ++column) {
      _stream << matrix[row][column];
      if (column != matrix.column_count() - 1U) {
        _stream << _delimiter;
      }
    }
    _stream << "\n";
  }

  _stream.flush();
}
}
}
