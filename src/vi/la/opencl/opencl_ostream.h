#ifndef __vinn__opencl_ostream__
#define __vinn__opencl_ostream__

#include <ostream>

namespace cl {
class Device;
}

std::ostream& operator<<(std::ostream& stream, const cl::Device& device);

#endif
