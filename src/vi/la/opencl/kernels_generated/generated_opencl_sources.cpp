#include "generated_opencl_sources.h"
#include <string>
#include "activation_functions_cl__source.h"
#include "convolution_cl__source.h"
#include "matrix_cl__source.h"
namespace vi {
namespace la {
namespace opencl_generated {
std::map<std::string, vi::la::opencl::source> paths_to_sources() {
  std::map<std::string, vi::la::opencl::source> mapping;
  const char* original_path;
  size_t data_length(0U);
  const char* data;

  vi::la::opencl_generated::activation_functions_cl__source(&original_path, &data, data_length);
  mapping.insert(std::pair<std::string, vi::la::opencl::source>(
      original_path, vi::la::opencl::source(data, data_length)));

  vi::la::opencl_generated::convolution_cl__source(&original_path, &data, data_length);
  mapping.insert(std::pair<std::string, vi::la::opencl::source>(
      original_path, vi::la::opencl::source(data, data_length)));

  vi::la::opencl_generated::matrix_cl__source(&original_path, &data, data_length);
  mapping.insert(std::pair<std::string, vi::la::opencl::source>(
      original_path, vi::la::opencl::source(data, data_length)));

  return mapping;
}
}
}
}
