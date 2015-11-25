%module(directors="1") vinnpy
%{
#define SWIG_FILE_WITH_INIT
#include <vi/la.h>
#include <vi/nn.h>
#include <vi/io/model.h>
#include <list>
#include <vector>
%}

%include "std_except.i"
%include "std_list.i"
%include "std_pair.i"
%include "std_shared_ptr.i"
%include "std_string.i"
%include "std_vector.i"
%include "numpy.i"
%init %{
  import_array();
%}

// include type information in doc strings
%feature("autodoc", "3");
// allow python subclasses of training_callback
%feature("director") vi::nn::training_callback;

%shared_ptr(vi::nn::activation_function);
%shared_ptr(vi::nn::sigmoid_activation);
%shared_ptr(vi::nn::softmax_activation);
%shared_ptr(vi::nn::hyperbolic_tangent);
%shared_ptr(vi::nn::linear_activation);

%shared_ptr(vi::nn::layer);

%template(matrix_vector) std::vector<vi::la::matrix>;
%template(matrix_matrix_pair) std::pair<vi::la::matrix, vi::la::matrix>;
%template(float_matrix_vector_pair) std::pair<float, std::vector<vi::la::matrix> >;
%template(size_t_pair) std::pair<size_t, size_t>;

%ignore vi::la::matrix::operator[];
%ignore vi::la::operator<<;
%ignore vi::nn::operator<<;

%extend vi::la::matrix {
  %rename(concatenate) operator<<;

  // support Python len operator
  size_t __len__() const {
    return self->row_count();
  }

  // support row access as NumPy arrays
  void __getitem__(size_t row, float** data, int* length) throw(std::out_of_range) {
    *data   = (*self)[row];
    *length = self->column_count();
  }

  // support numpy style access using python tuples
  float __getitem__(std::pair<size_t, size_t> row_and_column) throw(std::out_of_range) {
    float *row = self->operator[](row_and_column.first);

    if (row_and_column.second >= self->column_count()) {
      std::ostringstream details;
      details << "Column index: " << row_and_column.second << " out of range:[0," << self->column_count() - 1 << "]";
      throw std::out_of_range(details.str());
    }
    return row[row_and_column.second];
  }
}

// map rows returned from matrix::__getitem__ as numpy array views
%apply (float** ARGOUTVIEW_ARRAY1, int* DIM1) { (float**data, int* length) }

// allow passing numpy arrays to matrix constructor
%apply (float* IN_ARRAY2, int DIM1, int DIM2) { (const float* values, size_t rows, size_t columns) }


// Linear Algebra
%include <vi/la/context.h>
%include <vi/la/matrix.h>
%include <vi/la/opencl/opencl_context.h>
%include <vi/la/cpu/cpu_context.h>

// Neural Networks
%include <vi/nn/running_average.h>
%include <vi/nn/activation_function.h>
%include <vi/nn/trainer.h>
%include <vi/nn/batch_gradient_descent.h>
%include <vi/nn/confusion_table.h>
%include <vi/nn/cost_function.h>
%include <vi/nn/l2_regularizer.h>
%include <vi/nn/label_map.h>
%include <vi/nn/layer.h>
%include <vi/nn/minibatch_gradient_descent.h>
%include <vi/nn/network.h>
%include <vi/nn/result_measurements.h>

// Serialization/Deserialization
%include <vi/io/model.h>
