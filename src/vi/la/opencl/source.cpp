#include "source.h"

#include <cstring>

namespace vi {
namespace la {
namespace opencl {

source::source() : _data(nullptr), _length(0U) {}

source::source(const std::string& text_source) : _data(nullptr), _length(0U) {
  _length = text_source.length() + 1;
  _data = new char[_length];
  text_source.copy(_data, text_source.length());
  _data[text_source.length()] = '\0';
}

source::source(const char* data, size_t length) : _data(nullptr), _length(0U) {
  _length = length;
  _data = new char[_length];
  memcpy(_data, data, _length * sizeof(char));
}

source::~source() { delete[] _data; }

source::source(const source& other) {
  _data = new char[other.length()];
  _length = other.length();
  memcpy((void*)_data, other.data(), other.length() * sizeof(char));
}

source::source(source&& other) : _data(nullptr), _length(0U) { *this = std::move(other); }

source& source::operator=(const source& other) {
  if (this != &other) {
    delete[] _data;

    _data = new char[other.length()];
    _length = other.length();
    memcpy((void*)_data, other.data(), other.length() * sizeof(char));
  }
  return *this;
}

source& source::operator=(source&& other) {
  if (this != &other) {
    delete[] _data;

    _data = other._data;
    _length = other._length;
    other._data = nullptr;
    other._length = 0U;
  }
  return *this;
}

const char* source::data() const { return _data; }

size_t source::length() const { return _length; }
}
}
}
