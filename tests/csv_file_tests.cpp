#include "test.h"
#include "vi/io/csv_file.h"

#include <fstream>
#include <sstream>

using namespace std;

class csv_file_tests : public ::testing::TestWithParam<vi::la::context*> {};
INSTANTIATE_TEST_CASE_P(context, csv_file_tests, ::testing::ValuesIn(test::all_contexts()));

TEST_P(csv_file_tests, load_valid_file_succeeds) {
  const string matrix_path(test::fixture_path("valid.csv"));
  fstream matrix_file(matrix_path.c_str());
  vi::io::csv_file csv(matrix_file);
  vi::la::matrix m(*GetParam(), {{0.0}});
  EXPECT_NO_THROW(csv.load(m));
}

TEST_P(csv_file_tests, parse_valid_file_succeeds) {
  const string matrix_path(test::fixture_path("valid.csv"));
  fstream matrix_file(matrix_path.c_str());
  vi::io::csv_file csv(matrix_file);
  vi::la::matrix m(*GetParam(), {{0.0}});
  csv.load(m);

  vi::la::matrix expected(*GetParam(),
                          {{0.0, 0.1, 0.2, 0.3}, {1.0, 1.1, 1.2, 1.3}, {2.0, 2.1, 2.2, 2.3}});
  EXPECT_MATRIX_EQ(expected, m);
}

TEST_P(csv_file_tests, can_specify_delimiter) {
  const string matrix_path(test::fixture_path("valid.tsv"));
  fstream matrix_file(matrix_path.c_str());
  vi::io::csv_file csv(matrix_file, '\t');
  vi::la::matrix m(*GetParam(), {{0.0}});
  csv.load(m);

  vi::la::matrix expected(*GetParam(),
                          {{0.0, 0.1, 0.2, 0.3}, {1.0, 1.1, 1.2, 1.3}, {2.0, 2.1, 2.2, 2.3}});
  EXPECT_MATRIX_EQ(expected, m);
}

TEST_P(csv_file_tests, can_include_header) {
  const string matrix_path(test::fixture_path("valid_with_header.csv"));
  fstream matrix_file(matrix_path.c_str());
  vi::io::csv_file csv(matrix_file);
  vi::la::matrix m(*GetParam(), {{0.0}});
  std::vector<std::string> header;
  csv.load(m, header);

  std::vector<std::string> expected_header = {"col0", "col1", "col2", "col3"};
  vi::la::matrix expected(*GetParam(),
                          {{0.0, 0.1, 0.2, 0.3}, {1.0, 1.1, 1.2, 1.3}, {2.0, 2.1, 2.2, 2.3}});
  EXPECT_MATRIX_EQ(expected, m);
  EXPECT_EQ(expected_header, header);
}

TEST_P(csv_file_tests, load_invalid_file_path_fails) {
  fstream matrix_file("nowhere");
  vi::io::csv_file csv(matrix_file);
  vi::la::matrix m(*GetParam(), {{0.0}});
  EXPECT_THROW(csv.load(m), std::exception);
}

TEST_P(csv_file_tests, load_in_valid_string_fails) {
  stringstream stream;
  stream << "";
  vi::io::csv_file csv(stream);
  vi::la::matrix m(*GetParam(), {{0.0}});
  EXPECT_THROW(csv.load(m), std::exception);
}

TEST_P(csv_file_tests, load_valid_string_succeeds) {
  stringstream stream;
  stream << "1,2,3\n4,5,6\n7,8,9";
  vi::io::csv_file csv(stream);
  vi::la::matrix m(*GetParam(), {{0.0}});
  EXPECT_NO_THROW(csv.load(m));
}

TEST_P(csv_file_tests, parsing_valid_string_succeeds) {
  stringstream stream;
  stream << "1,2,3\n4,5,6\n7,8,9\n";
  vi::io::csv_file csv(stream);
  vi::la::matrix m(*GetParam(), {{0.0}});
  csv.load(m);

  vi::la::matrix expected(*GetParam(), {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
  EXPECT_MATRIX_EQ(expected, m);
}

TEST_P(csv_file_tests, storing_matrix_succeeds) {
  stringstream output;
  vi::io::csv_file csv(output);
  vi::la::matrix m(*GetParam(), {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
  EXPECT_NO_THROW(csv.store(m));
  const string expected_string = "1,2,3\n4,5,6\n7,8,9\n";
  EXPECT_EQ(expected_string, output.str());
}

TEST_P(csv_file_tests, storing_matrix_and_header_succeeds) {
  stringstream output;
  vi::io::csv_file csv(output);
  vi::la::matrix m(*GetParam(), {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
  std::vector<std::string> header = {"col0", "col1", "col2"};
  EXPECT_NO_THROW(csv.store(m, header));
  const string expected_string = "col0,col1,col2\n1,2,3\n4,5,6\n7,8,9\n";
  EXPECT_EQ(expected_string, output.str());
}
