#ifndef __vinn__libsvm__file__
#define __vinn__libsvm__file__

#include <vi/la/matrix.h>
#include <iostream>
#include <vector>

namespace vi {
namespace io {

/// Load sparse label and feature matrices stored in the libsvm format
/// http://www.csie.ntu.edu.tw/~cjlin/libsvm/
class libsvm_file {
public:
  /// Construct with an input stream
  libsvm_file(std::iostream& stream);

  /// Load labels and features
  std::pair<vi::la::matrix, vi::la::matrix> load_labels_and_features(vi::la::context& context,
                                                                     size_t max_feature_count = 0U);

private:
  typedef std::pair<size_t, float> sparse_entry;
  typedef std::pair<std::vector<float>, std::vector<sparse_entry>> labels_and_features_row;
  typedef std::vector<labels_and_features_row> labels_and_features_vector;

  void fill_from_sparse_data(labels_and_features_vector sparse_data, vi::la::matrix& labels,
                             vi::la::matrix& features);
  labels_and_features_vector parse_contents(size_t& largest_label_index,
                                            size_t& largest_feature_index,
                                            const size_t max_feature_index);
  labels_and_features_row parse_row(const std::string& line, size_t& largest_feature_index,
                                    const size_t max_feature_index) const;

  std::iostream& _stream;
};
}
}

#endif
