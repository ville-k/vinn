#ifndef __vinn__opencl_builder__
#define __vinn__opencl_builder__

#include <vi/la/opencl/build_result.h>
#include <vi/la/opencl/source.h>
#include <vi/la/opencl/source_loader.h>

#include <set>
#include <list>
#include <vector>

namespace cl {
class Context;
}

namespace vi {
namespace la {
namespace opencl {

/// Builder knows how to build opencl programs using the specified
/// source files, options and device requirements
class builder {
public:
  builder(source_loader& loader);

  void add_extension_requirements(const std::vector<std::string>& required);
  void add_source_paths(const std::vector<std::string>& paths);
  void add_build_options(const std::vector<std::string>& options);

  /// return build result containing the opencl program on success
  build_result build(cl::Context& context);
  bool can_build(cl::Context& context) const;

private:
  std::list<source> load_sources() const;
  std::string combine_build_options() const;

  bool compiler_available(cl::Context& context) const;
  bool supports_all_required_extensions(cl::Context& context) const;

  source_loader& _loader;
  std::set<std::string> _required_extensions;
  std::set<std::string> _source_paths;
  std::set<std::string> _build_options;
};
}
}
}

#endif
