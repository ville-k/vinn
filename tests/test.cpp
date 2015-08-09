#include "test.h"
#include "vi/la/cpu/cpu_context.h"
#include "vi/la/matrix.h"
#include "vi/la/opencl/opencl_context.h"
#include "vi/la/opencl/opencl_builder.h"
#include "vi/la/opencl/disk_source_loader.h"
#include "vi/la/opencl/opencl_ostream.h"

#include <algorithm>

namespace test {

std::string fixture_path(const std::string fixture_file) {
  const std::string fixture_dir_path(std::string(SRCROOT) + "/tests/fixtures");
  return fixture_dir_path + "/" + fixture_file;
}

std::vector<vi::la::context*> all_contexts() {
  static std::vector<vi::la::context*> contexts = {};

  if (contexts.size() == 0) {
    contexts.push_back(new vi::la::cpu_context());

    std::vector<cl_device_id> device_ids = vi::la::opencl_context::supported_devices();
    for (cl_device_id device_id : device_ids) {
      contexts.push_back(new vi::la::opencl_context({device_id}));
    }
  }

  return contexts;
}
}

::testing::AssertionResult AssertMatricesEqual(const char* expected_expression,
                                               const char* actual_expression,
                                               const vi::la::matrix& expected,
                                               const vi::la::matrix& actual) {
  if (actual.size() != expected.size()) {
    return ::testing::AssertionFailure()
           << "Matrices " << expected_expression << " and " << actual_expression
           << " are not equal - dimensions differ: " << expected.row_count() << "x"
           << expected.column_count() << " != " << actual.row_count() << "x"
           << actual.column_count();
  }

  for (size_t m = 0U; m < expected.row_count(); ++m) {
    for (size_t n = 0U; n < expected.column_count(); ++n) {
      // using googletest internals, but this class has not changed in years
      const ::testing::internal::FloatingPoint<float> lhs(expected[m][n]), rhs(actual[m][n]);
      if (!lhs.AlmostEquals(rhs)) {
        return ::testing::AssertionFailure()
               << "Matrices \"" << expected_expression << "\" and \"" << actual_expression
               << "\" are not equal - elements at " << m << "," << n
               << " differ: " << expected[m][n] << " != " << actual[m][n];
      }
    }
  }

  return ::testing::AssertionSuccess();
}
