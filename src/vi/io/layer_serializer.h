#ifndef __vinn__layer_serializer__
#define __vinn__layer_serializer__

#include <vi/io/serializer.h>
#include <vi/nn/layer.h>

namespace vi {
namespace io {
class layer_serializer : public serializer {
public:
  layer_serializer(const vi::nn::layer& layer);
  virtual void serialize(boost::property_tree::ptree& layer_node);

private:
  const vi::nn::layer& layer_;
};
}
}

#endif
