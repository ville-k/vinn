#ifndef __vinn__memory_source_loader__
#define __vinn__memory_source_loader__

#include <vi/la/opencl/source_loader.h>
#include <map>

namespace vi {
namespace la {
namespace opencl {

/// Load OpenCL sources stored inside an application/library
class memory_source_loader : public source_loader {
public:
  memory_source_loader(const std::map<std::string, source>& path_to_source);

  virtual source load(const std::string& relative_path);
  virtual bool can_load(const std::string& relative_path) const;

private:
  std::map<std::string, source> _path_to_source;
};
}
}
}

#endif
