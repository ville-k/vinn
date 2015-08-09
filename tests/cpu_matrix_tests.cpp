#include "test.h"
#include "vi/la/cpu/cpu_matrix.h"

TEST(cpu_matrix, constructs_with_nullptr) {
  vi::la::cpu_context context;
  vi::la::cpu::matrix* test = nullptr;
  const size_t ROW_COUNT = 4;
  const size_t COLUMN_COUNT = 4;
  EXPECT_NO_THROW(test = new vi::la::cpu::matrix(context, ROW_COUNT, COLUMN_COUNT, nullptr));
  EXPECT_EQ(ROW_COUNT, test->row_count());
  EXPECT_EQ(COLUMN_COUNT, test->column_count());
  EXPECT_NO_THROW(delete test);
}

TEST(cpu_matrix, constructs_with_initial_values) {
  vi::la::cpu_context context;
  vi::la::cpu::matrix* test = nullptr;
  const size_t ROW_COUNT = 4;
  const size_t COLUMN_COUNT = 4;
  float initial_values[4 * 4];
  for (size_t i = 0U; i < ROW_COUNT * COLUMN_COUNT; ++i) {
    initial_values[i] = static_cast<float>(i);
  }
  EXPECT_NO_THROW(test = new vi::la::cpu::matrix(context, 4, 4, initial_values));
  EXPECT_EQ(ROW_COUNT, test->row_count());
  EXPECT_EQ(COLUMN_COUNT, test->column_count());
  for (size_t i = 0U; i < ROW_COUNT * COLUMN_COUNT; ++i) {
    EXPECT_FLOAT_EQ(initial_values[i], test->raw_data()[i]);
  }
  EXPECT_NO_THROW(delete test);
}
