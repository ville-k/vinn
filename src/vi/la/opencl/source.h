#ifndef __vinn__source__
#define __vinn__source__

#include <string>

namespace vi {
namespace la {
namespace opencl {

/// Container resource for kernel source data (textual or binary)
class source {
public:
  source();
  source(const std::string& text_source);
  source(const char* data, size_t length);
  ~source();

  source(const source& other);
  source(source&& other);
  source& operator=(const source& other);
  source& operator=(source&& other);

  const char* data() const;
  size_t length() const;

private:
  char* _data;
  size_t _length;
};
}
}
}

#endif
