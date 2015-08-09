#include "vi/io/libsvm_file.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cassert>

namespace vi {
namespace io {

libsvm_file::libsvm_file(std::iostream& stream) : _stream(stream) {}

std::pair<vi::la::matrix, vi::la::matrix>
libsvm_file::load_labels_and_features(vi::la::context& context, size_t max_feature_count) {
  _stream.exceptions(std::iostream::badbit);
  _stream.seekg(0U);

  size_t largest_feature_index(0U);
  size_t largest_label_index(0U);
  labels_and_features_vector rows =
      parse_contents(largest_label_index, largest_feature_index, max_feature_count);

  if (max_feature_count != 0U) {
    largest_feature_index = max_feature_count - 1U;
  }
  size_t row_count = rows.size();

  vi::la::matrix labels(context, row_count, largest_label_index + 1);
  vi::la::matrix features(context, row_count, largest_feature_index + 1);
  fill_from_sparse_data(rows, labels, features);

  return std::make_pair(labels, features);
}

void libsvm_file::fill_from_sparse_data(labels_and_features_vector sparse_data,
                                        vi::la::matrix& labels, vi::la::matrix& features) {
  for (size_t m = 0U; m < sparse_data.size(); ++m) {
    auto row = sparse_data[m];
    for (size_t i = 0U; i < row.first.size(); ++i) {
      labels[m][i] = row.first[i];
    }

    for (const sparse_entry& entry : row.second) {
      size_t column = entry.first;
      float value = entry.second;
      features[m][column] = value;
    }
  }
}

libsvm_file::labels_and_features_vector
libsvm_file::parse_contents(size_t& largest_label_index, size_t& largest_feature_index,
                            const size_t max_feature_count) {
  labels_and_features_vector rows;

  std::string line;
  while (std::getline(_stream, line, '\n')) {
    size_t rows_largest_index(0U);
    labels_and_features_row row = parse_row(line, rows_largest_index, max_feature_count);
    largest_feature_index = std::max(largest_feature_index, rows_largest_index);
    largest_label_index = std::max(largest_label_index, row.first.size() - 1);
    rows.push_back(row);
  }

  return rows;
}

libsvm_file::labels_and_features_row libsvm_file::parse_row(const std::string& line,
                                                            size_t& largest_feature_index,
                                                            const size_t max_feature_count) const {
  // chop off trailing comments
  std::string row;
  std::istringstream line_stream(line);
  std::getline(line_stream, row, '#');

  std::istringstream row_stream(row);
  std::string element;
  labels_and_features_row labels_and_features;
  bool label_section(true);
  while (std::getline(row_stream, element, ' ')) {
    if (label_section) {
      std::istringstream label_stream(element);
      std::string label_string;
      while (std::getline(label_stream, label_string, ',')) {
        float label = std::stof(label_string);
        labels_and_features.first.push_back(label);
      }
      label_section = false;
    } else {
      std::istringstream value_stream(element);
      std::string value_string;
      bool is_index(true);
      size_t index(0);
      float value(0.0);

      while (std::getline(value_stream, value_string, ':')) {
        if (is_index) {
          index = std::stoull(value_string) - 1;
          is_index = false;
        } else {
          value = std::stof(value_string);
        }
      }

      if (max_feature_count != 0U && index >= max_feature_count) {
        // user limited number of features to load
        continue;
      }
      largest_feature_index = std::max(largest_feature_index, index);
      sparse_entry feature(index, value);
      labels_and_features.second.push_back(feature);
    }
  }

  return labels_and_features;
}
}
}
