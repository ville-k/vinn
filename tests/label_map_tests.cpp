#include "test.h"
#include "vi/nn/label_map.h"
#include "vi/la/matrix.h"

#include <vector>

using namespace vi::la;
using vi::nn::label_map;
using vi::nn::unknown_label_exception;
using std::vector;

class label_map_tests : public ::testing::TestWithParam<context*> {

protected:
  matrix identity_matrix(size_t rows, size_t columns) {
    matrix identity(*GetParam(), rows, columns, 0.0);

    for (size_t m = 0U; m < rows; ++m) {
      for (size_t n = 0U; n < columns; ++n) {
        if (m == n) {
          identity[m][n] = 1.0;
        }
      }
    }
    return identity;
  }
};

INSTANTIATE_TEST_CASE_P(context, label_map_tests, ::testing::ValuesIn(test::all_contexts()));

TEST_P(label_map_tests, creates_mapping_from_label_count) {
  const size_t label_count(3U);
  label_map map(label_count);

  const matrix activations = identity_matrix(label_count, label_count);
  const matrix labels(map.activations_to_labels(activations));
  EXPECT_EQ(1U, labels.column_count());
  EXPECT_EQ(label_count, labels.row_count());

  for (size_t m = 0U; m < label_count; ++m) {
    float expected_label(static_cast<float>(m));
    EXPECT_FLOAT_EQ(expected_label, labels[m][0U]);
  }
}

TEST_P(label_map_tests, creates_mapping_from_labels) {
  vector<int> labels = {42, 7};
  label_map map(labels);

  const size_t label_count(labels.size());
  const matrix activations = identity_matrix(label_count, label_count);

  const matrix mapped_labels(map.activations_to_labels(activations));
  EXPECT_EQ(1U, mapped_labels.column_count());
  EXPECT_EQ(label_count, mapped_labels.row_count());

  for (size_t m = 0U; m < label_count; ++m) {
    float expected_label(static_cast<float>(labels[m]));
    EXPECT_FLOAT_EQ(expected_label, mapped_labels[m][0U]);
  }
}

TEST_P(label_map_tests, maps_labels_to_vectors) {
  vector<int> labels = {42, 7};
  label_map map(labels);

  const matrix label_matrix(*GetParam(), labels.size(), 1U);
  for (size_t m = 0; m < label_matrix.row_count(); ++m) {
    label_matrix[m][0U] = labels[m];
  }

  const matrix mapped_activations(map.labels_to_activations(label_matrix));
  EXPECT_EQ(2U, mapped_activations.column_count());
  EXPECT_EQ(label_matrix.row_count(), mapped_activations.row_count());

  for (size_t m = 0U; m < labels.size(); ++m) {
    for (size_t n = 0U; n < labels.size(); ++n) {
      if (n == m) {
        EXPECT_FLOAT_EQ(1.0, mapped_activations[m][n]);
      } else {
        EXPECT_FLOAT_EQ(0.0, mapped_activations[m][n]);
      }
    }
  }
}

TEST_P(label_map_tests, providing_duplicate_labels_throws) {
  vector<int> labels = {7, 42, 7};
  EXPECT_THROW(label_map map(labels), unknown_label_exception);
}

TEST_P(label_map_tests, mapping_unknown_label_throws) {
  vector<int> labels = {42, 7};
  label_map map(labels);

  matrix label_matrix(*GetParam(), 1, 3);
  label_matrix[0U][2U] = 1;
  EXPECT_THROW(map.labels_to_activations(label_matrix), unknown_label_exception);
}
