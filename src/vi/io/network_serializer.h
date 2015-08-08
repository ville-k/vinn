#ifndef __vinn__network_serializer__
#define __vinn__network_serializer__

#include <vi/io/serializer.h>
#include <vi/nn/network.h>

namespace vi {
namespace io {
class network_serializer : public serializer {
public:
  network_serializer(const vi::nn::network& network);
  virtual void serialize(boost::property_tree::ptree& network_node);

private:
  const vi::nn::network& network_;
};
}
}

#endif /* defined(__vinn__network_serializer__) */
