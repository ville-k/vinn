#ifndef __vinn__disk_source_loader__
#define __vinn__disk_source_loader__

#include <vi/la/opencl/source_loader.h>

namespace vi {
namespace la {
namespace opencl {

/// Load OpenCL sources from the filesystem
class disk_source_loader : public vi::la::opencl::source_loader {
public:
  disk_source_loader(const std::string& source_root);

  virtual source load(const std::string& relative_path);
  virtual bool can_load(const std::string& relative_path) const;

private:
  bool is_directory(const std::string& path) const;
  bool is_file(const std::string& path) const;
  std::string _source_root;
};
}
}
}

#endif
