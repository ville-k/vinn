#ifndef __vinn__result_measurements__
#define __vinn__result_measurements__

#include <ostream>
#include <vector>

#include <vi/nn/confusion_table.h>
#include <vi/la/context.h>
#include <vi/la/matrix.h>

namespace vi {
namespace nn {

/// Performance measures for classification
/// For a good overview see:
/// http://rali.iro.umontreal.ca/rali/sites/default/files/publis/SokolovaLapalme-JIPM09.pdf

/// Measures for multi-class classification
class result_measurements {
public:
  result_measurements(vi::la::context& context, const std::vector<int>& labels);

  void add_results(const std::vector<int>& expected, const std::vector<int>& actual);
  void add_results(const vi::la::matrix& expected, const vi::la::matrix& actual);

  float accuracy() const;

  /// Macro averages - treat all classes the same
  float average_accuracy() const;
  float error_rate() const;
  float precision() const;
  float recall() const;
  float fscore(float beta = 1.0) const;

  /// Micro averages - favor large classes
  float micro_precision() const;
  float micro_recall() const;
  float micro_fscore(float beta = 1.0) const;

  vi::la::matrix confusion_matrix() const;

  confusion_table confusion_table_for_label(int label) const;

  const std::vector<int>& labels() const;

private:
  size_t label_index_for_label(int label) const;
  void update_confusion_matrix(const std::vector<int>& expected, const std::vector<int>& actual);

  vi::la::matrix _confusion_matrix;
  const std::vector<int> _labels;
};

std::ostream& operator<<(std::ostream& os, const result_measurements& measurements);
}
}

#endif
