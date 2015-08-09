#include "test.h"
#include "vi/nn/result_measurements.h"
#include "vi/la/matrix.h"

#include <vector>

using std::vector;
using vi::la::matrix;
using vi::nn::result_measurements;

class result_measurements_tests : public ::testing::TestWithParam<vi::la::context*> {
protected:
  enum Animals { Cat, Dog };

  virtual void SetUp() {
    labels = {Cat, Dog};
    expected = {Cat, Dog};
  }

  vector<int> labels;
  vector<int> expected;
};

INSTANTIATE_TEST_CASE_P(context, result_measurements_tests,
                        ::testing::ValuesIn(test::all_contexts()));

TEST_P(result_measurements_tests, none_correct) {
  vector<int> actual = {Dog, Cat};

  result_measurements measures(*GetParam(), labels);
  measures.add_results(expected, actual);
  EXPECT_FLOAT_EQ(0.0, measures.accuracy());
  EXPECT_FLOAT_EQ(0.0, measures.average_accuracy());
  EXPECT_FLOAT_EQ(1.0, measures.error_rate());
  EXPECT_FLOAT_EQ(0.0, measures.precision());
  EXPECT_FLOAT_EQ(0.0, measures.recall());
  EXPECT_FLOAT_EQ(0.0, measures.fscore());
}

TEST_P(result_measurements_tests, half_correct) {
  vector<int> actual = {Cat, Cat};

  result_measurements measures(*GetParam(), labels);
  measures.add_results(expected, actual);
  EXPECT_FLOAT_EQ(0.5, measures.accuracy());
  EXPECT_FLOAT_EQ(0.5, measures.average_accuracy());
  EXPECT_FLOAT_EQ(0.5, measures.error_rate());
  EXPECT_FLOAT_EQ(0.25, measures.precision());
  EXPECT_FLOAT_EQ(0.5, measures.recall());
  EXPECT_FLOAT_EQ(1.0 / 3.0, measures.fscore());
}

TEST_P(result_measurements_tests, all_correct) {
  vector<int> actual = {Cat, Dog};

  result_measurements measures(*GetParam(), labels);
  measures.add_results(expected, actual);
  EXPECT_FLOAT_EQ(1.0, measures.accuracy());
  EXPECT_FLOAT_EQ(1.0, measures.average_accuracy());
  EXPECT_FLOAT_EQ(0.0, measures.error_rate());
  EXPECT_FLOAT_EQ(1.0, measures.precision());
  EXPECT_FLOAT_EQ(1.0, measures.recall());
  EXPECT_FLOAT_EQ(1.0, measures.fscore());
}

TEST_P(result_measurements_tests, matrix_results_all_correct) {
  vi::la::matrix actual_matrix(*GetParam(), 2, 1);
  actual_matrix[0][0] = Cat;
  actual_matrix[1][0] = Dog;
  vi::la::matrix expected_matrix(*GetParam(), 2, 1);
  expected_matrix[0][0] = Cat;
  expected_matrix[1][0] = Dog;

  result_measurements measures(*GetParam(), labels);
  measures.add_results(expected_matrix, actual_matrix);
  EXPECT_FLOAT_EQ(1.0, measures.accuracy());
  EXPECT_FLOAT_EQ(1.0, measures.average_accuracy());
  EXPECT_FLOAT_EQ(0.0, measures.error_rate());
  EXPECT_FLOAT_EQ(1.0, measures.precision());
  EXPECT_FLOAT_EQ(1.0, measures.recall());
  EXPECT_FLOAT_EQ(1.0, measures.fscore());
}
