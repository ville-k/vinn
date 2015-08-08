#ifndef __vinn__layer_deserializer__
#define __vinn__layer_deserializer__

#include <vi/nn/layer.h>
#include <vi/io/deserializer.h>

namespace vi {
namespace io {
class layer_deserializer : public deserializer {
public:
  layer_deserializer(vi::nn::layer& layer);

  virtual void deserialize(const boost::property_tree::ptree& layer_node);

private:
  vi::nn::layer& layer_;
};
}
}

#endif
