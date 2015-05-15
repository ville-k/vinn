#include "vi/la/opencl/opencl_builder.h"
#include "vi/la/opencl/opencl_ostream.h"

#include <algorithm>
#include <CL/cl.hpp>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace vi {
namespace la {
namespace opencl {

builder::builder(source_loader& loader) : _loader(loader) {}

void builder::add_extension_requirements(const std::vector<std::string>& required) {
  _required_extensions.insert(required.begin(), required.end());
}

void builder::add_source_paths(const std::vector<std::string>& paths) {
  for (const std::string& relative_path : paths) {
    if (!_loader.can_load(relative_path)) {
      throw std::invalid_argument("invalid source path: " + relative_path);
    }
    _source_paths.insert(relative_path);
  }
}

void builder::add_build_options(const std::vector<std::string>& options) {
  _build_options.insert(options.begin(), options.end());
}

bool builder::can_build(cl::Context& context) const {
  if (!compiler_available(context)) {
    return false;
  }
  if (!supports_all_required_extensions(context)) {
    return false;
  }
  return true;
}

build_result builder::build(cl::Context& context) {
  build_result result;
  result.set_success(false);

  if (!can_build(context)) {
    result.set_log("build context does not support all required features");
    return result;
  }

  // list owns and frees the source memory
  std::list<source> loaded_sources = load_sources();
  cl::Program::Sources sources;
  for (const auto& source : loaded_sources) {
    // source lenght should not include null termination
    sources.push_back({source.data(), source.length() - 1U});
  }
  cl::Program program(context, sources);

  std::vector<cl::Device> context_devices = context.getInfo<CL_CONTEXT_DEVICES>();
  std::ostringstream log_stream;
  try {
    program.build(context_devices, combine_build_options().c_str());
    result.set_program(program);
    result.set_success(true);
    log_stream << "Build succeeded" << std::endl;
  } catch (cl::Error& error) {
    result.set_success(false);
    log_stream << "Build failed" << std::endl;
    log_stream << "Error: " << error.what() << std::endl;
    log_stream << "Error code: " << error.err() << std::endl;
  }

  for (size_t device_index = 0; device_index < context_devices.size(); ++device_index) {
    cl::Device& device = context_devices[device_index];
    log_stream << device;
    log_stream << "Build log: " << std::endl;
    log_stream << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
  }
  result.set_log(log_stream.str());
  return result;
}

std::list<source> builder::load_sources() const {
  std::list<source> sources;
  for (const auto& path : _source_paths) {
    sources.push_back(_loader.load(path));
  }
  return sources;
}

std::string builder::combine_build_options() const {
  std::string options;
  for (const std::string& option : _build_options) {
    options += option + " ";
  }
  return options;
}

bool builder::compiler_available(cl::Context& context) const {
  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
  for (cl::Device& device : devices) {
    cl_bool available = device.getInfo<CL_DEVICE_COMPILER_AVAILABLE>();
    if (!available) {
      return false;
    }
  }
  return true;
}

bool builder::supports_all_required_extensions(cl::Context& context) const {
  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

  for (cl::Device& device : devices) {
    std::string extensions_string = device.getInfo<CL_DEVICE_EXTENSIONS>();
    std::istringstream iss(extensions_string);
    std::set<std::string> available_extensions;
    std::copy(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>(),
              std::inserter(available_extensions, available_extensions.begin()));

    std::set<std::string> matching_extensions;
    std::set_intersection(available_extensions.begin(), available_extensions.end(),
                          _required_extensions.begin(), _required_extensions.end(),
                          std::inserter(matching_extensions, matching_extensions.begin()));
    if (matching_extensions != _required_extensions) {
      return false;
    }
  }

  return true;
}
}
}
}
