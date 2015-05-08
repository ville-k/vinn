#ifndef __vinn__build_result__
#define __vinn__build_result__

#include <CL/cl.hpp>

namespace vi {
namespace la {
namespace opencl {

/// Build product of opencl::builder
class build_result {
public:
  bool success() const;
  void set_success(bool success);

  cl::Program program();
  void set_program(cl::Program& program);

  std::string log() const;
  void set_log(const std::string& log);

private:
  cl::Program _program;
  bool _success;
  std::string _log;
};

}
}
}

#endif

