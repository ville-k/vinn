#include "test.h"
#include "vi/io/libsvm_file.h"

#include <string>
#include <sstream>
#include <iostream>

using vi::io::libsvm_file;

class libsvm_file_tests : public ::testing::TestWithParam<vi::la::context*> {};
INSTANTIATE_TEST_CASE_P(context, libsvm_file_tests, ::testing::ValuesIn(test::all_contexts()));

TEST_P(libsvm_file_tests, constructs) {
  std::stringstream stream;
  EXPECT_NO_THROW(libsvm_file file(stream));
}

TEST_P(libsvm_file_tests, load_single_label_and_features) {
  const char* contents = R"Contents(5 1:1 2:2 159:159 160:160
2 1:0.1 2:0.22 3:0.333 159:0.159 160:0.160)Contents";

  std::stringstream stream(contents);
  libsvm_file file(stream);

  auto labels_and_features = file.load_labels_and_features(*GetParam());

  EXPECT_EQ(2U, labels_and_features.first.row_count());
  EXPECT_EQ(1U, labels_and_features.first.column_count());
  EXPECT_EQ(5.0, labels_and_features.first[0][0]);
  EXPECT_EQ(2.0, labels_and_features.first[1][0]);

  EXPECT_EQ(2U, labels_and_features.second.row_count());
  EXPECT_EQ(160U, labels_and_features.second.column_count());

  EXPECT_EQ(1.0f, labels_and_features.second[0][0]);
  EXPECT_EQ(2.0f, labels_and_features.second[0][1]);
  EXPECT_EQ(159.0f, labels_and_features.second[0][158]);
  EXPECT_EQ(160.0f, labels_and_features.second[0][159]);

  EXPECT_EQ(0.1f, labels_and_features.second[1][0]);
  EXPECT_EQ(0.22f, labels_and_features.second[1][1]);
  EXPECT_EQ(0.333f, labels_and_features.second[1][2]);
  EXPECT_EQ(0.159f, labels_and_features.second[1][158]);
  EXPECT_EQ(0.160f, labels_and_features.second[1][159]);
}

TEST_P(libsvm_file_tests, load_with_max_feature_count) {
  const char* contents = R"Contents(5 1:1 2:2 159:159 160:160
2 1:0.1 2:0.22 3:0.333 159:0.159 160:0.160)Contents";

  std::stringstream stream(contents);
  libsvm_file file(stream);

  auto labels_and_features = file.load_labels_and_features(*GetParam(), 159U);

  EXPECT_EQ(2U, labels_and_features.first.row_count());
  EXPECT_EQ(1U, labels_and_features.first.column_count());
  EXPECT_EQ(5.0f, labels_and_features.first[0][0]);
  EXPECT_EQ(2.0f, labels_and_features.first[1][0]);

  EXPECT_EQ(2U, labels_and_features.second.row_count());
  EXPECT_EQ(159U, labels_and_features.second.column_count());

  EXPECT_EQ(1.0f, labels_and_features.second[0][0]);
  EXPECT_EQ(2.0f, labels_and_features.second[0][1]);
  EXPECT_EQ(159.0, labels_and_features.second[0][158]);

  EXPECT_EQ(0.1f, labels_and_features.second[1][0]);
  EXPECT_EQ(0.22f, labels_and_features.second[1][1]);
  EXPECT_EQ(0.333f, labels_and_features.second[1][2]);
  EXPECT_EQ(0.159f, labels_and_features.second[1][158]);
}

TEST_P(libsvm_file_tests, load_multiple_labels_and_features) {
  const char* contents = R"Contents(5,2 1:1 2:2 159:159 160:160
2,1,3 1:0.1 2:0.22 3:0.333 159:0.159 160:0.160)Contents";

  std::stringstream stream(contents);
  libsvm_file file(stream);

  auto labels_and_features = file.load_labels_and_features(*GetParam());

  EXPECT_EQ(2U, labels_and_features.first.row_count());
  EXPECT_EQ(3U, labels_and_features.first.column_count());

  EXPECT_EQ(5.0f, labels_and_features.first[0][0]);
  EXPECT_EQ(2.0f, labels_and_features.first[1][0]);

  EXPECT_EQ(2U, labels_and_features.second.row_count());
  EXPECT_EQ(160U, labels_and_features.second.column_count());

  EXPECT_EQ(1.0f, labels_and_features.second[0][0]);
  EXPECT_EQ(2.0f, labels_and_features.second[0][1]);
  EXPECT_EQ(159.0f, labels_and_features.second[0][158]);
  EXPECT_EQ(160.0f, labels_and_features.second[0][159]);

  EXPECT_EQ(0.1f, labels_and_features.second[1][0]);
  EXPECT_EQ(0.22f, labels_and_features.second[1][1]);
  EXPECT_EQ(0.333f, labels_and_features.second[1][2]);
  EXPECT_EQ(0.159f, labels_and_features.second[1][158]);
  EXPECT_EQ(0.160f, labels_and_features.second[1][159]);
}

TEST_P(libsvm_file_tests, load_data_with_comments) {
  const char* contents = R"Contents(5 1:1 2:2 159:159 160:160 # this is data
2 1:0.1 2:0.22 3:0.333 159:0.159 160:0.160)Contents";

  std::stringstream stream(contents);
  libsvm_file file(stream);

  auto labels_and_features = file.load_labels_and_features(*GetParam(), 160);

  EXPECT_EQ(2U, labels_and_features.first.row_count());
  EXPECT_EQ(1U, labels_and_features.first.column_count());
  EXPECT_EQ(5.0f, labels_and_features.first[0][0]);
  EXPECT_EQ(2.0f, labels_and_features.first[1][0]);

  EXPECT_EQ(2U, labels_and_features.second.row_count());
  EXPECT_EQ(160U, labels_and_features.second.column_count());

  EXPECT_EQ(1.0f, labels_and_features.second[0][0]);
  EXPECT_EQ(2.0f, labels_and_features.second[0][1]);
  EXPECT_EQ(159.0f, labels_and_features.second[0][158]);
  EXPECT_EQ(160.0f, labels_and_features.second[0][159]);

  EXPECT_EQ(0.1f, labels_and_features.second[1][0]);
  EXPECT_EQ(0.22f, labels_and_features.second[1][1]);
  EXPECT_EQ(0.333f, labels_and_features.second[1][2]);
  EXPECT_EQ(0.159f, labels_and_features.second[1][158]);
  EXPECT_EQ(0.160f, labels_and_features.second[1][159]);
}
