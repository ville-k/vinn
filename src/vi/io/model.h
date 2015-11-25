#ifndef __vinn__model__
#define __vinn__model__

#include <vi/la/context.h>
#include <vi/nn/network.h>
#include <string>
#include <boost/filesystem/path.hpp>

namespace vi {
namespace io {

class model {
public:
  class exception : public std::runtime_error {
  public:
    exception(const std::string& text) : runtime_error(text) {}
  };

  model(const std::string& path);

  void load(vi::nn::network& network, vi::la::context& context);
  void store(const vi::nn::network& network);

private:
  const std::string path_;

  boost::filesystem::path model_dir_path() const;
  boost::filesystem::path model_description_path() const;

  boost::filesystem::path model_data_dir_path() const;
  boost::filesystem::path layer_path(size_t layer_index) const;
};
}
}

#endif
