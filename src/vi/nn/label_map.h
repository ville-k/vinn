#ifndef __vinn__label_map__
#define __vinn__label_map__

#include <iostream>
#include <vector>
#include <stdexcept>

#include <vi/la/matrix.h>

namespace vi {
namespace nn {

/// Encountered a label that the map does not know about
class unknown_label_exception : public std::runtime_error {
public:
  unknown_label_exception(const std::string& error) : std::runtime_error(error) {}
};

/// Map activation units to labels and vice versa
class label_map {
public:
  /// label map with one-to one mapping of activation units
  /// [0, label_count[  -> [0, label_count[
  label_map(const size_t label_count);
  /// [0, labels.size() - 1] -> [labels[0], labels[last]]
  label_map(std::vector<int> labels);

  /// map highest probability activation with corresponding label
  vi::la::matrix activations_to_labels(const vi::la::matrix& activations) const;

  /// map labels into activation vectors based on mapping configuration
  vi::la::matrix labels_to_activations(const vi::la::matrix& labels) const;

  /// return all available labels
  const std::vector<int>& labels() const;

private:
  void create_mappings(const std::vector<int>& labels);

  // vector index of a label indicates active unit
  std::vector<int> _active_unit_to_label;
};
}
}

#endif
