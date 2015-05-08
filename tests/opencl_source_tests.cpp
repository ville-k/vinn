#include "test.h"
#include "vi/la/opencl/source.h"

TEST(opencl_source, constructs_with_string) {
  std::string source_string = "0123456789";

  vi::la::opencl::source source(source_string);
  EXPECT_EQ(11U, source.length());
  EXPECT_EQ('\0', source.data()[10]);
  EXPECT_EQ(0, memcmp(source_string.c_str(), source.data(), 11));
}

TEST(opencl_source, constructs_with_data_and_length) {
  const char* data = "0123456789";
  const size_t length = 11U;
  vi::la::opencl::source source(data, length);

  EXPECT_EQ(11U, source.length());
  EXPECT_EQ('\0', source.data()[10]);
  EXPECT_EQ(0, memcmp(data, source.data(), 11));
}
