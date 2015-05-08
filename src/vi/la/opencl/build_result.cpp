#include "vi/la/opencl/build_result.h"

namespace vi {
namespace la {
namespace opencl {

bool build_result::success() const { return _success; }

void build_result::set_success(bool success) { _success = success; }

cl::Program build_result::program() { return _program; }

void build_result::set_program(cl::Program& program) { _program = program; }

std::string build_result::log() const { return _log; }

void build_result::set_log(const std::string& log) { _log = log; }

}
}
}

