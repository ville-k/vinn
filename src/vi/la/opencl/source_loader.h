#ifndef __vinn__source_loader__
#define __vinn__source_loader__

#include <vi/la/opencl/source.h>

namespace vi {
namespace la {
namespace opencl {

/// Interface that classes loading OpenCL sources must conform to
class source_loader {
public:
  virtual source load(const std::string& relative_path) = 0;
  virtual bool can_load(const std::string& relative_path) const = 0;
};
}
}
}

#endif
