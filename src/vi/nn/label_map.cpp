#include "vi/nn/label_map.h"

#include <cmath>
#include <limits>
#include <set>
#include <sstream>

namespace vi {
namespace nn {

label_map::label_map(const size_t label_count) {
  std::vector<int> labels;
  for (size_t label = 0U; label < label_count; ++label) {
    labels.push_back(static_cast<int>(label));
  }
  create_mappings(labels);
}

label_map::label_map(std::vector<int> labels) { create_mappings(labels); }

vi::la::matrix label_map::activations_to_labels(const vi::la::matrix& activations) const {
  vi::la::matrix labels(activations.owning_context(), activations.row_count(), 1U);

  for (size_t m = 0U; m < activations.row_count(); ++m) {
    float current_max_prob(0.0);
    size_t label_index(0U);
    for (size_t n = 0U; n < activations.column_count(); ++n) {
      float probablity(activations[m][n]);
      if (probablity > current_max_prob) {
        label_index = n;
        current_max_prob = probablity;
      }
    }
    labels[m][0U] = _active_unit_to_label[label_index];
  }

  return labels;
}

vi::la::matrix label_map::labels_to_activations(const vi::la::matrix& labels) const {
  vi::la::matrix vectors(labels.owning_context(), labels.row_count(), _active_unit_to_label.size(),
                         0.0);

  for (size_t m = 0U; m < vectors.row_count(); ++m) {
    bool label_found(false);
    float label = labels[m][0U];
    for (size_t n = 0U; n < vectors.column_count(); ++n) {
      if (std::fabs(_active_unit_to_label[n] - label) < std::numeric_limits<float>::epsilon()) {
        vectors[m][n] = 1.0;
        label_found = true;
        break;
      }
    }

    if (!label_found) {
      std::ostringstream details;
      details << "Row " << m << " contains an unknown label: ";
      details << label << std::endl;
      throw unknown_label_exception(details.str());
    }
  }

  return vectors;
}

const std::vector<int>& label_map::labels() const { return _active_unit_to_label; }

void label_map::create_mappings(const std::vector<int>& labels) {
  std::set<int> duplicates;
  for (size_t i = 0U; i < labels.size(); ++i) {
    int label = labels[i];

    if (duplicates.find(label) != duplicates.end()) {
      std::ostringstream details;
      details << "Duplicate label specified: " << label << std::endl;
      throw unknown_label_exception(details.str());
    } else {
      duplicates.insert(label);
    }
  }

  _active_unit_to_label = labels;
}
}
}
