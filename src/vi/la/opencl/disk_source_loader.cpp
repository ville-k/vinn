#include "vi/la/opencl/disk_source_loader.h"
#include <sys/stat.h>
#include <fstream>
#include <stdexcept>

namespace vi {
namespace la {
namespace opencl {

disk_source_loader::disk_source_loader(const std::string& source_root) : _source_root(source_root) {
  if (!is_directory(_source_root)) {
    throw std::invalid_argument("invalid source root: " + _source_root);
  }
}

bool disk_source_loader::is_directory(const std::string& path) const {
  struct stat sb;
  if (stat(path.c_str(), &sb) == -1) {
    return false;
  }
  if (!S_ISDIR(sb.st_mode)) {
    return false;
  }
  return true;
}

bool disk_source_loader::is_file(const std::string& path) const {
  struct stat sb;
  if (stat(path.c_str(), &sb) == -1) {
    return false;
  }
  if (!S_ISREG(sb.st_mode)) {
    return false;
  }
  return true;
}

source disk_source_loader::load(const std::string& relative_path) {
  std::string absolute_path = _source_root + "/" + relative_path;
  if (!is_file(absolute_path)) {
    throw std::invalid_argument("invalid source path: " + absolute_path);
  }

  std::ifstream stream(absolute_path);
  std::string contents((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  return source(contents);
}

bool disk_source_loader::can_load(const std::string& relative_path) const {
  std::string absolute_path = _source_root + "/" + relative_path;
  return is_file(absolute_path);
}
}
}
}
