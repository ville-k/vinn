#include "test.h"
#include "vi/la/opencl/disk_source_loader.h"
#include "vi/la/opencl/memory_source_loader.h"

TEST(opencl_disk_loader_tests, construction_succeeds_for_valid_source_root) {
  EXPECT_NO_THROW(vi::la::opencl::disk_source_loader loader("/"));
}

TEST(opencl_disk_loader_tests, construction_fails_for_invalid_source_root) {
  EXPECT_THROW(vi::la::opencl::disk_source_loader loader("/no/such/location/exists"),
               std::invalid_argument);
}

TEST(opencl_disk_loader_tests, loading_from_valid_path_does_not_throw) {
  vi::la::opencl::disk_source_loader loader(std::string(SRCROOT));
  EXPECT_NO_THROW(loader.load("tests/fixtures/kernel.cl"));
}

TEST(opencl_disk_loader_tests, loading_from_valid_path_loads_source) {
  vi::la::opencl::disk_source_loader loader(std::string(SRCROOT));
  vi::la::opencl::source source = loader.load("tests/fixtures/kernel.cl");
  EXPECT_NE(nullptr, source.data());
  EXPECT_LT(0U, source.length());

  std::string expected_source = "\n\
__kernel void add_numbers(__global float * a, float b) {\n\
  float c = a[0] + b;\n\
}\n\n";

  EXPECT_EQ(expected_source, std::string(source.data()));
}

TEST(opencl_disk_loader_tests, loading_from_invalid_paths_fails) {
  vi::la::opencl::disk_source_loader loader(std::string(SRCROOT));
  EXPECT_THROW(loader.load("tests/fixtures/does_not_exist.cl"), std::invalid_argument);
}

TEST(memory_source_loader, loading_from_valid_path_loads_source) {
  std::string expected_source = "\
__kernel void add_numbers(__global float * a, float b) {\n\
    float c = a[0] + b;\n\
}";
  std::string source_path = "test/kernel.cl";
  std::map<std::string, vi::la::opencl::source> source_map;
  source_map[source_path] = vi::la::opencl::source(expected_source);
  vi::la::opencl::memory_source_loader loader(source_map);
  vi::la::opencl::source source = loader.load(source_path);
  EXPECT_EQ(expected_source, std::string(source.data()));
}
