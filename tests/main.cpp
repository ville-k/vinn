#include <gtest/gtest.h>

int main(int argc, const char* argv[]) {
  testing::InitGoogleTest(&argc, (wchar_t**)argv);
  return RUN_ALL_TESTS();
}
