#include "vi/nn/confusion_table.h"
#include <cmath>

namespace vi {
namespace nn {

confusion_table::confusion_table(size_t true_positives, size_t false_negatives,
                                 size_t false_positives, size_t true_negatives)
    : _true_positives(true_positives), _true_negatives(true_negatives),
      _false_positives(false_positives), _false_negatives(false_negatives) {}

confusion_table::confusion_table(const confusion_table& other) { *this = other; }

confusion_table& confusion_table::operator=(const confusion_table& other) {
  _true_positives = other._true_positives;
  _false_negatives = other._false_negatives;
  _false_positives = other._false_positives;
  _true_negatives = other._true_negatives;

  return *this;
}

size_t confusion_table::true_positives() const { return _true_positives; }

size_t confusion_table::true_negatives() const { return _true_negatives; }

size_t confusion_table::false_positives() const { return _false_positives; }

size_t confusion_table::false_negatives() const { return _false_negatives; }

float confusion_table::tp() const { return static_cast<float>(_true_positives); }

float confusion_table::tn() const { return static_cast<float>(_true_negatives); }

float confusion_table::fp() const { return static_cast<float>(_false_positives); }

float confusion_table::fn() const { return static_cast<float>(_false_negatives); }

float confusion_table::accuracy() const {
  const float denominator(results());
  if (denominator > 0.0) {
    return (tp() + tn()) / denominator;
  }
  return 0.0;
}

float confusion_table::error_rate() const {
  const float denominator(results());
  if (denominator > 0.0) {
    return (fp() + fn()) / denominator;
  }
  return 0.0;
}

float confusion_table::precision() const {
  const float denominator(tp() + fp());
  if (denominator > 0) {
    return tp() / denominator;
  }
  return 0.0;
}

float confusion_table::recall() const {
  const float denominator(tp() + fn());
  if (denominator > 0.0) {
    return tp() / denominator;
  }
  return 0.0;
}

float confusion_table::fscore(float beta) const {
  const float beta_squared(std::pow(beta, 2));
  const float denominator((beta_squared + 1) * tp() + beta_squared * fn() + fp());
  if (denominator > 0.0) {
    return ((beta_squared + 1) * tp()) / denominator;
  }
  return 0.0;
}

float confusion_table::specificity() const {
  const float denominator(fp() + tn());
  if (denominator > 0) {
    return tn() / denominator;
  }
  return 0.0;
}

float confusion_table::auc() const { return 0.5 * (recall() + specificity()); }

float confusion_table::results() const { return tp() + tn() + fp() + fn(); }

std::ostream& operator<<(std::ostream& os, const confusion_table& t) {
  os << "Accuracy:  " << t.accuracy() << std::endl;
  os << "Precision: " << t.precision() << std::endl;
  os << "Recall:    " << t.recall() << std::endl;
  os << "FScore:    " << t.fscore() << std::endl;
  return os;
}
}
}
