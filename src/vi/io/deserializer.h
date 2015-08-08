#ifndef __vinn__deserializer__
#define __vinn__deserializer__

#include <boost/property_tree/ptree.hpp>

namespace vi {
namespace io {
class deserializer {
public:
  class exception : public std::runtime_error {
  public:
    exception(const std::string& text) : runtime_error(text) {}
  };

  virtual void deserialize(const boost::property_tree::ptree& layer_node) = 0;
};
}
}

#endif
