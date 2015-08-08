#include "test.h"
#include "vi/la/opencl/disk_source_loader.h"
#include "vi/la/opencl/memory_source_loader.h"
#include "vi/la/opencl/opencl_builder.h"

#include <CL/cl.hpp>
#include <vector>

using namespace vi::la::opencl;
class opencl_builder_tests : public ::testing::TestWithParam<cl::Context> {
  void SetUp() { source_root = SRCROOT; }

protected:
  std::string source_root;
};

std::vector<cl::Context> testing_contexts() {
  std::vector<cl::Context> contexts;

  std::vector<cl::Platform> platforms;
  try {
    cl::Platform::get(&platforms);
  } catch (cl::Error& error) {
    // can throw when no drivers are installed - ignore
  }

  for (auto& platform : platforms) {
    std::vector<cl::Device> available_devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &available_devices);
    for (cl::Device& device : available_devices) {
      contexts.push_back(cl::Context({device}));
    }
  }

  return contexts;
}

INSTANTIATE_TEST_CASE_P(opencl_context, opencl_builder_tests,
                        ::testing::ValuesIn(testing_contexts()));

TEST_P(opencl_builder_tests, construction_succeeds_for_valid_source_root) {
  disk_source_loader loader(source_root);
  EXPECT_NO_THROW(builder builder(loader));
}

TEST_P(opencl_builder_tests, adding_valid_source_paths_succeeds) {
  disk_source_loader loader(source_root);
  builder builder(loader);
  builder.add_source_paths({"tests/fixtures/kernel.cl"});
}

TEST_P(opencl_builder_tests, adding_invalid_source_paths_fails) {
  disk_source_loader loader(source_root);
  builder builder(loader);
  EXPECT_THROW(builder.add_source_paths({"tests/fixtures/does_not_exist.cl"}),
               std::invalid_argument);
}

TEST_P(opencl_builder_tests, building_valid_kernel_succeeds) {
  disk_source_loader loader(source_root);
  builder builder(loader);
  builder.add_source_paths({"tests/fixtures/kernel.cl"});

  cl::Context ctx = GetParam();
  build_result result = builder.build(ctx);
  EXPECT_TRUE(result.success());
  EXPECT_NO_THROW({ cl::Kernel kernel(result.program(), "add_numbers"); });
}

TEST_P(opencl_builder_tests, building_valid_kernels_from_different_sources_succeeds) {
  disk_source_loader loader(source_root);
  builder builder(loader);
  builder.add_source_paths({"tests/fixtures/kernel.cl", "tests/fixtures/another_kernel.cl"});

  cl::Context ctx = GetParam();
  build_result result = builder.build(ctx);
  EXPECT_TRUE(result.success());
  EXPECT_NO_THROW({ cl::Kernel kernel(result.program(), "add_numbers"); });
  EXPECT_NO_THROW({ cl::Kernel kernel(result.program(), "subtract_numbers"); });
}

TEST_P(opencl_builder_tests, building_valid_kernels_from_different_memory_sources_succeeds) {
  std::map<std::string, vi::la::opencl::source> source_map;

  const char* kernel_source = "\
__kernel void add_numbers(__global float * a, float b) {\n\
    float c = a[0] + b;\n\
}";
  const size_t kernel_source_length = 82 + 1;
  std::string kernel_source_path = "memory/fixtures/kernel.cl";
  source_map[kernel_source_path] = vi::la::opencl::source(kernel_source, kernel_source_length);

  const char* another_kernel_source = "\
__kernel void subtract_numbers(__global float * a, float b) {\n\
    float c = a[0] - b;\n\
}";
  const size_t another_kernel_source_length = 87 + 1;
  std::string another_kernel_source_path = "memory/fixtures/another_kernel.cl";
  source_map[another_kernel_source_path] =
      vi::la::opencl::source(another_kernel_source, another_kernel_source_length);

  memory_source_loader loader(source_map);

  builder builder(loader);
  builder.add_source_paths({kernel_source_path, another_kernel_source_path});

  cl::Context ctx = GetParam();
  build_result result = builder.build(ctx);
  EXPECT_TRUE(result.success());
  EXPECT_NO_THROW({ cl::Kernel kernel(result.program(), "add_numbers"); });

  EXPECT_NO_THROW({ cl::Kernel kernel(result.program(), "subtract_numbers"); });
}

TEST_P(opencl_builder_tests, building_invalid_kernel_fails) {
  disk_source_loader loader(source_root);
  builder builder(loader);
  builder.add_source_paths({"tests/fixtures/invalid_kernel.cl"});

  cl::Context ctx = GetParam();
  EXPECT_FALSE(builder.build(ctx).success());
}

TEST_P(opencl_builder_tests, adding_preprocessor_macros_succeeds) {
  disk_source_loader loader(source_root);
  builder builder(loader);
  builder.add_source_paths({"tests/fixtures/kernel.cl"});

  builder.add_build_options({"-DDOUBLE_SUPPORT_AVAILABLE", "-DNONSENSE"});

  cl::Context ctx = GetParam();
  build_result result = builder.build(ctx);
  EXPECT_TRUE(result.success());
  // TODO : test the macros took effect
}

TEST_P(opencl_builder_tests, building_with_available_extensions_succeed) {
  disk_source_loader loader(source_root);
  builder builder(loader);
  builder.add_extension_requirements({"cl_khr_fp64"});
  builder.add_source_paths({"tests/fixtures/kernel.cl"});
  cl::Context ctx = GetParam();
  EXPECT_NO_THROW(builder.build(ctx));
}

TEST_P(opencl_builder_tests, building_with_unavailable_extensions_fails) {
  disk_source_loader loader(source_root);
  builder builder(loader);
  builder.add_extension_requirements({"made_up_extension"});
  builder.add_source_paths({"tests/fixtures/kernel.cl"});
  cl::Context ctx = GetParam();
  EXPECT_FALSE(builder.build(ctx).success());
}
