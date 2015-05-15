#include "vi/la/opencl/memory_source_loader.h"

namespace vi {
namespace la {
namespace opencl {

memory_source_loader::memory_source_loader(const std::map<std::string, source>& path_to_source)
    : _path_to_source(path_to_source) {}

source memory_source_loader::load(const std::string& relative_path) {
  return _path_to_source[relative_path];
}

bool memory_source_loader::can_load(const std::string& relative_path) const {
  return _path_to_source.find(relative_path) != _path_to_source.end();
}
}
}
}
