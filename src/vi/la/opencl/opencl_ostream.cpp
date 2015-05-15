#include "vi/la/opencl/opencl_ostream.h"
#include <CL/cl.hpp>

std::ostream& operator<<(std::ostream& stream, const cl::Device& device) {
  stream << device.getInfo<CL_DEVICE_NAME>() << std::endl;
  stream << "  - max compute units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
  stream << "  - max work group size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()
         << std::endl;
  stream << "  - max work item sizes: " << std::endl;
  size_t dimension = 0;
  for (const auto& size : device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()) {
    stream << "      " << dimension << ": " << size << std::endl;
    ++dimension;
  }

  stream << "  - max work item dimensions: " << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>()
         << std::endl;
  stream << "  - max clock freq: " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
  stream << "  - max alloc size: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() / (1024 * 1024)
         << " MB" << std::endl;
  stream << "  - global mem size: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024)
         << " MB" << std::endl;

  stream << "  - local mem size: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << " B"
         << std::endl;
  stream << "  - constant mem size: " << device.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>()
         << " B" << std::endl;

  stream << "OpenCL: version: " << device.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;
  stream << "device version: " << device.getInfo<CL_DEVICE_VERSION>() << std::endl;

  return stream;
}
