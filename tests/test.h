#ifndef __vinn__test__
#define __vinn__test__

#include "vi/la/context.h"

#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace test {

std::string fixture_path(const std::string fixture_file);

std::vector<vi::la::context*> all_contexts();
}

::testing::AssertionResult AssertMatricesEqual(const char* expected_expression,
                                               const char* actual_expression,
                                               const vi::la::matrix& expected,
                                               const vi::la::matrix& actual);

#define EXPECT_MATRIX_EQ(expected, actual)                                                         \
  EXPECT_PRED_FORMAT2(AssertMatricesEqual, expected, actual)

#endif
