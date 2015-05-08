#ifndef __vinn__csv__file__
#define __vinn__csv__file__

#include <iostream>
#include <vi/la/matrix.h>

namespace vi {
namespace io {

/// Load matrices stored in CSV format
class csv_file {
public:
  csv_file(std::iostream& stream);
  virtual ~csv_file();

  vi::la::matrix load(vi::la::context& context);

private:
  std::iostream& _stream;
};

}
}

#endif

