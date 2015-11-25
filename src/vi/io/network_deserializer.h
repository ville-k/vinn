#ifndef __vinn__network_deserializer__
#define __vinn__network_deserializer__

#include <vi/la/context.h>
#include <vi/nn/network.h>
#include <vi/io/deserializer.h>

namespace vi {
namespace io {
class network_deserializer {
public:
  network_deserializer(vi::nn::network& network);

  virtual void deserialize(const boost::property_tree::ptree& network_node,
                           vi::la::context& context);

private:
  vi::nn::network& network_;
};
}
}

#endif
