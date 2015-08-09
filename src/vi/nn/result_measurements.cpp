#include "vi/nn/result_measurements.h"
#include <cmath>
#include <limits>

using vi::la::matrix;

namespace {

/// create a std::vector representation of column vector/matrix
std::vector<int> column_vector_to_vector(const vi::la::matrix& column_vector_labels) {
  std::vector<int> vector;
  vector.reserve(column_vector_labels.row_count());
  for (size_t m = 0U; m < column_vector_labels.row_count(); ++m) {
    int value = column_vector_labels[m][0U];
    vector.push_back(value);
  }

  return vector;
}
}

namespace vi {
namespace nn {

result_measurements::result_measurements(vi::la::context& context, const std::vector<int>& labels)
    : _confusion_matrix(context, labels.size(), labels.size(), 0.0), _labels(labels) {}

void result_measurements::add_results(const std::vector<int>& expected,
                                      const std::vector<int>& actual) {
  update_confusion_matrix(expected, actual);
}

void result_measurements::add_results(const vi::la::matrix& expected,
                                      const vi::la::matrix& actual) {
  add_results(column_vector_to_vector(expected), column_vector_to_vector(actual));
}

void result_measurements::update_confusion_matrix(const std::vector<int>& expected,
                                                  const std::vector<int>& actual) {
  for (size_t m = 0U; m < actual.size(); ++m) {
    const size_t actual_index(label_index_for_label(actual[m]));
    const size_t expected_index(label_index_for_label(expected[m]));

    _confusion_matrix[expected_index][actual_index] += 1.0;
  }
}

float result_measurements::accuracy() const {
  float total(0.0);
  float correct(0.0);

  for (size_t m = 0U; m < _confusion_matrix.row_count(); ++m) {
    for (size_t n = 0U; n < _confusion_matrix.column_count(); ++n) {
      float count(_confusion_matrix[m][n]);
      total += count;
      if (m == n) {
        correct += count;
      }
    }
  }

  return correct / total;
}

float result_measurements::average_accuracy() const {
  float all_accuracies(0.0);
  for (int label : _labels) {
    all_accuracies += confusion_table_for_label(label).accuracy();
  }

  return all_accuracies / _labels.size();
}

float result_measurements::error_rate() const {
  float total(0.0);
  for (int label : _labels) {
    total += confusion_table_for_label(label).error_rate();
  }

  return total / _labels.size();
}

float result_measurements::precision() const {
  float total(0.0);
  for (int label : _labels) {
    total += confusion_table_for_label(label).precision();
  }

  return total / _labels.size();
}

float result_measurements::recall() const {
  float total(0.0);
  for (int label : _labels) {
    total += confusion_table_for_label(label).recall();
  }

  return total / _labels.size();
}

float result_measurements::fscore(float beta) const {
  const float beta_squared(std::pow(beta, 2));
  const float p(precision());
  const float r(recall());
  if (p > 0.0 || r > 0.0) {
    return ((beta_squared + 1.0) * p * r) / (beta_squared * p + r);
  }
  return 0.0;
}

float result_measurements::micro_precision() const {
  float total_tp(0.0);
  float total_fp(0.0);
  for (int label : _labels) {
    confusion_table table = confusion_table_for_label(label);
    total_tp += table.true_positives();
    total_fp += table.false_positives();
  }

  return total_tp / (total_tp + total_fp);
}

float result_measurements::micro_recall() const {
  float total_tp(0.0);
  float total_fn(0.0);
  for (int label : _labels) {
    confusion_table table = confusion_table_for_label(label);
    total_tp += table.true_positives();
    total_fn += table.false_negatives();
  }

  return total_tp / (total_tp + total_fn);
}

float result_measurements::micro_fscore(float beta) const {
  const float mp(micro_precision());
  const float mr(micro_recall());
  const float beta_squared = std::pow(beta, 2);

  return ((beta_squared + 1) * mp * mr) / (beta_squared * mp + mr);
}

size_t result_measurements::label_index_for_label(int label) const {
  size_t label_index = 0U;
  for (size_t i = 0U; i < _labels.size(); ++i) {
    if (label == _labels[i]) {
      label_index = i;
      break;
    }
  }

  return label_index;
}

vi::la::matrix result_measurements::confusion_matrix() const { return _confusion_matrix; }

confusion_table result_measurements::confusion_table_for_label(int label) const {
  float true_positives(0.0);
  float true_negatives(0.0);
  float false_positives(0.0);
  float false_negatives(0.0);
  size_t label_index = label_index_for_label(label);

  for (size_t m = 0U; m < _confusion_matrix.row_count(); ++m) {
    for (size_t n = 0U; n < _confusion_matrix.column_count(); ++n) {
      float predictions(_confusion_matrix[m][n]);

      if (m == label_index) {
        if (n == label_index) {
          true_positives = predictions;
        } else {
          false_negatives += predictions;
        }
      } else {
        if (n == label_index) {
          false_positives += predictions;
        } else {
          true_negatives += predictions;
        }
      }
    }
  }

  return confusion_table(true_positives, false_negatives, false_positives, true_negatives);
}

const std::vector<int>& result_measurements::labels() const { return _labels; }

std::ostream& operator<<(std::ostream& os, const result_measurements& m) {
  os << "Performance measures: " << std::endl;
  os << "Accuracy:           " << m.accuracy() << std::endl;
  os << "Average accuracy:   " << m.average_accuracy() << std::endl;
  os << "Average error rate: " << m.error_rate() << std::endl;
  os << "Average precision:  " << m.precision() << std::endl;
  os << "Average recall:     " << m.recall() << std::endl;
  os << "Average fscore:     " << m.fscore() << std::endl;
  os << "Micro precision:    " << m.micro_precision() << std::endl;
  os << "Micro recall:       " << m.micro_recall() << std::endl;
  os << "Micro fscore:       " << m.micro_fscore() << std::endl;
  os << std::endl;
  os << "Confusion matrix: " << std::endl;
  os << m.confusion_matrix() << std::endl;

  os << "Per class measures" << std::endl;
  for (int label : m.labels()) {
    os << "Label:     " << label << std::endl;
    os << m.confusion_table_for_label(label);
    os << std::endl;
  }

  return os;
}
}
}
