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
  result_measurements(vi::la::context& context,
                      const std::vector<long>& labels);

  void add_results(const std::vector<long>& expected,
                   const std::vector<long>& actual);
  void add_results(const vi::la::matrix& expected,
                   const vi::la::matrix& actual);

  double accuracy() const;

  /// Macro averages - treat all classes the same
  double average_accuracy() const;
  double error_rate() const;
  double precision() const;
  double recall() const;
  double fscore(double beta = 1.0) const;

  /// Micro averages - favor large classes
  double micro_precision() const;
  double micro_recall() const;
  double micro_fscore(double beta = 1.0) const;

  vi::la::matrix confusion_matrix() const;

  confusion_table confusion_table_for_label(long label) const;

  const std::vector<long>& labels() const;

private:
  size_t label_index_for_label(long label) const;
  void update_confusion_matrix(const std::vector<long>& expected,
                               const std::vector<long>& actual);

  vi::la::matrix _confusion_matrix;
  const std::vector<long> _labels;
};

std::ostream& operator<<(std::ostream& os,
                         const result_measurements& measurements);

}
}

#endif

