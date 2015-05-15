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

double confusion_table::tp() const { return static_cast<double>(_true_positives); }

double confusion_table::tn() const { return static_cast<double>(_true_negatives); }

double confusion_table::fp() const { return static_cast<double>(_false_positives); }

double confusion_table::fn() const { return static_cast<double>(_false_negatives); }

double confusion_table::accuracy() const {
  const double denominator(results());
  if (denominator > 0.0) {
    return (tp() + tn()) / denominator;
  }
  return 0.0;
}

double confusion_table::error_rate() const {
  const double denominator(results());
  if (denominator > 0.0) {
    return (fp() + fn()) / denominator;
  }
  return 0.0;
}

double confusion_table::precision() const {
  const double denominator(tp() + fp());
  if (denominator > 0) {
    return tp() / denominator;
  }
  return 0.0;
}

double confusion_table::recall() const {
  const double denominator(tp() + fn());
  if (denominator > 0.0) {
    return tp() / denominator;
  }
  return 0.0;
}

double confusion_table::fscore(double beta) const {
  const double beta_squared(std::pow(beta, 2));
  const double denominator((beta_squared + 1) * tp() + beta_squared * fn() + fp());
  if (denominator > 0.0) {
    return ((beta_squared + 1) * tp()) / denominator;
  }
  return 0.0;
}

double confusion_table::specificity() const {
  const double denominator(fp() + tn());
  if (denominator > 0) {
    return tn() / denominator;
  }
  return 0.0;
}

double confusion_table::auc() const { return 0.5 * (recall() + specificity()); }

double confusion_table::results() const { return tp() + tn() + fp() + fn(); }

std::ostream& operator<<(std::ostream& os, const confusion_table& t) {
  os << "Accuracy:  " << t.accuracy() << std::endl;
  os << "Precision: " << t.precision() << std::endl;
  os << "Recall:    " << t.recall() << std::endl;
  os << "FScore:    " << t.fscore() << std::endl;
  return os;
}
}
}
