#ifndef __vinn__confusion_table__
#define __vinn__confusion_table__

#include <stddef.h>
#include <ostream>

namespace vi {
namespace nn {

/// Measures for binary classification
class confusion_table {
public:
  confusion_table(size_t true_positives, size_t false_negatives, size_t false_positives,
                  size_t true_negatives);
  confusion_table(const confusion_table& other);
  confusion_table& operator=(const confusion_table& other);

  size_t true_positives() const;
  size_t true_negatives() const;
  size_t false_positives() const;
  size_t false_negatives() const;

  float accuracy() const;
  float error_rate() const;
  float precision() const;
  float recall() const;
  float fscore(float beta = 1.0) const;
  float specificity() const;
  float auc() const;

private:
  float results() const;
  float tp() const;
  float tn() const;
  float fp() const;
  float fn() const;

  size_t _true_positives;
  size_t _true_negatives;
  size_t _false_positives;
  size_t _false_negatives;
};

std::ostream& operator<<(std::ostream& os, const confusion_table& table);
}
}

#endif
