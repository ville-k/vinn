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
  EXPECT_NO_THROW(csv.load(*GetParam()));
}

TEST_P(csv_file_tests, parse_valid_file_succeeds) {
  const string matrix_path(test::fixture_path("valid.csv"));
  fstream matrix_file(matrix_path.c_str());
  vi::io::csv_file csv(matrix_file);
  vi::la::matrix m(csv.load(*GetParam()));

  vi::la::matrix expected(*GetParam(),
                          {{0.0, 0.1, 0.2, 0.3}, {1.0, 1.1, 1.2, 1.3}, {2.0, 2.1, 2.2, 2.3}});
  EXPECT_MATRIX_EQ(expected, m);
}

TEST_P(csv_file_tests, load_invalid_file_path_fails) {
  fstream matrix_file("nowhere");
  vi::io::csv_file csv(matrix_file);
  EXPECT_THROW(csv.load(*GetParam()), std::exception);
}

TEST_P(csv_file_tests, load_in_valid_string_fails) {
  stringstream stream;
  stream << "";
  vi::io::csv_file csv(stream);
  EXPECT_THROW(csv.load(*GetParam()), std::exception);
}

TEST_P(csv_file_tests, load_valid_string_succeeds) {
  stringstream stream;
  stream << "1,2,3\n4,5,6\n7,8,9";
  vi::io::csv_file csv(stream);
  EXPECT_NO_THROW(csv.load(*GetParam()));
}

TEST_P(csv_file_tests, parsing_valid_string_succeeds) {
  stringstream stream;
  stream << "1,2,3\n4,5,6\n7,8,9\n";
  vi::io::csv_file csv(stream);
  vi::la::matrix m(csv.load(*GetParam()));

  vi::la::matrix expected(*GetParam(), {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
  EXPECT_MATRIX_EQ(expected, m);
}
