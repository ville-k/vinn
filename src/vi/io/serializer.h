#ifndef __vinn__serializer__
#define __vinn__serializer__

#include <boost/property_tree/ptree.hpp>
#include <stdexcept>

namespace vi {
namespace io {
class serializer {
public:
  class exception : public std::runtime_error {
  public:
    exception(const std::string& text) : runtime_error(text) {}
  };

  virtual void serialize(boost::property_tree::ptree& layer_node) = 0;
};
}
}

#endif
