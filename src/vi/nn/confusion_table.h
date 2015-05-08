#ifndef __vinn__confusion_table__
#define __vinn__confusion_table__

#include <stddef.h>
#include <ostream>

namespace vi {
namespace nn {

/// Measures for binary classification
class confusion_table {
public:
  confusion_table(size_t true_positives, size_t false_negatives,
                  size_t false_positives, size_t true_negatives);
  confusion_table(const confusion_table& other);
  confusion_table& operator=(const confusion_table& other);

  size_t true_positives() const;
  size_t true_negatives() const;
  size_t false_positives() const;
  size_t false_negatives() const;

  double accuracy() const;
  double error_rate() const;
  double precision() const;
  double recall() const;
  double fscore(double beta = 1.0) const;
  double specificity() const;
  double auc() const;

private:
  double results() const;
  double tp() const;
  double tn() const;
  double fp() const;
  double fn() const;

  size_t _true_positives;
  size_t _true_negatives;
  size_t _false_positives;
  size_t _false_negatives;
};

std::ostream& operator<<(std::ostream& os, const confusion_table& table);

}
}

#endif

