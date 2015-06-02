#ifndef __vinn__csv__file__
#define __vinn__csv__file__

#include <iostream>
#include <vi/la/matrix.h>

namespace vi {
namespace io {

/// Load matrices stored in CSV format
class csv_file {
public:
  csv_file(std::iostream& stream, char delimiter = ',');
  virtual ~csv_file();

  void load(vi::la::matrix& matrix);
  void load(vi::la::matrix& matrix, std::vector<std::string>& header);

  void store(const vi::la::matrix& matrix);
  void store(const vi::la::matrix& matrix, std::vector<std::string>& header);

private:
  void load(vi::la::matrix& matrix, std::vector<std::string>* header);
  void parse_header(const std::string& line, std::vector<std::string>& header) const;
  void parse_row(const std::string& line, std::vector<double>& row) const;
  std::shared_ptr<double> make_buffer(const std::vector<std::vector<double>>& matrix_values,
                                      size_t max_columns) const;

  void store(const vi::la::matrix& matrix, std::vector<std::string>* header);

  char _delimiter;
  std::iostream& _stream;
};
}
}

#endif
