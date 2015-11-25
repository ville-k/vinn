#include "vi/la/opencl/opencl_context.h"

#include "vi/la/matrix.h"
#include "vi/la/opencl/disk_source_loader.h"
#include "vi/la/opencl/memory_source_loader.h"
#include "vi/la/opencl/opencl_builder.h"
#include "vi/la/opencl/opencl_matrix.h"
#include "vi/la/opencl/kernels_generated/generated_opencl_sources.h"

#include <cassert>
#include <CL/cl.hpp>
#include <cmath>

namespace vi {
namespace la {

opencl_context::opencl_context(const std::vector<cl_device_id>& device_ids) {
  std::vector<cl::Device> devices;
  for (cl_device_id device_id : device_ids) {
    devices.push_back(cl::Device(device_id));
  }
  _context = new cl::Context(devices);
  _command_queue = new cl::CommandQueue(*_context, devices[0]);
  load_kernels();
}

opencl_context::~opencl_context() {
  delete _command_queue;
  delete _context;
}

std::vector<cl_device_id> opencl_context::supported_devices(cl_device_type device_type) {
  std::vector<cl_device_id> supported_devices;
  vi::la::opencl::disk_source_loader loader("/");
  vi::la::opencl::builder builder(loader);
  builder.add_extension_requirements({"cl_khr_fp64"});

  std::vector<cl::Platform> platforms;
  try {
    cl::Platform::get(&platforms);
  } catch (cl::Error& error) {
    // can throw when no drivers are installed - ignore
  }

  for (auto& platform : platforms) {
    std::vector<cl::Device> available_devices;
    platform.getDevices(device_type, &available_devices);
    for (cl::Device& device : available_devices) {
      cl::Context context({device});
      if (builder.can_build(context)) {
        supported_devices.push_back(device());
      }
    }
  }

  return supported_devices;
}

void opencl_context::load_kernels() {
#if CONFIGURATION == Debug
  const std::string program_dir(std::string(SRCROOT) + "/src/vi/la/opencl/kernels");
  opencl::disk_source_loader loader(program_dir);
#else
  opencl::memory_source_loader loader(vi::la::opencl::paths_to_sources());
#endif
  opencl::builder builder(loader);
  // builder.add_build_options({"-DDOUBLE_SUPPORT_AVAILABLE"});
  builder.add_source_paths({"matrix.cl", "activation_functions.cl", "convolution.cl"});
  builder.add_extension_requirements({"cl_khr_fp64"});
  opencl::build_result result = builder.build(*_context);
  if (!result.success()) {
    throw std::runtime_error(result.log());
  }

  cl::Program program = result.program();
  _matrix_multiply_kernel = new cl::Kernel(program, "matrix_multiply");
  _matrix_scalar_multiply = new cl::Kernel(program, "matrix_scalar_multiply");
  _matrix_elementwise_multiply = new cl::Kernel(program, "matrix_elementwise_multiply");
  _matrix_add = new cl::Kernel(program, "matrix_add");
  _scalar_add = new cl::Kernel(program, "scalar_add");
  _matrix_subtract = new cl::Kernel(program, "matrix_subtract");
  _matrix_merge_kernel = new cl::Kernel(program, "matrix_merge");
  _matrix_transpose_kernel = new cl::Kernel(program, "matrix_transpose");
  _sum_rows = new cl::Kernel(program, "sum_rows");
  _sum_columns = new cl::Kernel(program, "sum_columns");
  _log = new cl::Kernel(program, "matrix_log");

  _matrix_sigmoid_kernel = new cl::Kernel(program, "matrix_sigmoid");
  _sigmoid_gradient = new cl::Kernel(program, "matrix_sigmoid_gradient");
  _hyperbolic_tangent = new cl::Kernel(program, "matrix_hyperbolic_tangent");
  _hyperbolic_tangent_gradient = new cl::Kernel(program, "matrix_hyperbolic_tangent_gradient");
  _matrix_softmax_exp_kernel = new cl::Kernel(program, "matrix_softmax_exp");
  _matrix_softmax_normalize_kernel = new cl::Kernel(program, "matrix_softmax_normalize");

  _convolve_2d = new cl::Kernel(program, "matrix_convolve_2d");
}

cl::Context& opencl_context::context() { return *_context; }

cl::CommandQueue& opencl_context::command_queue() { return *_command_queue; }

std::shared_ptr<vi::la::matrix_implementation>
opencl_context::implement_matrix(size_t rows, size_t columns, const float* initial_values) {
  return std::shared_ptr<vi::la::matrix_implementation>(
      new opencl::matrix(*this, rows, columns, initial_values));
}

void opencl_context::multiply(matrix& product, const matrix& operand_1, const matrix& operand_2) {
  opencl::matrix* product_impl = (opencl::matrix*)(product.implementation());
  opencl::matrix* operand_1_impl = (opencl::matrix*)(operand_1.implementation());
  opencl::matrix* operand_2_impl = (opencl::matrix*)(operand_2.implementation());

  _matrix_multiply_kernel->setArg(0, *product_impl->get());
  _matrix_multiply_kernel->setArg(1, *operand_1_impl->get());
  _matrix_multiply_kernel->setArg(2, *operand_2_impl->get());

  // operand_1: m x n
  // operand_2: n x k
  // product:   m x k

  _matrix_multiply_kernel->setArg(3, operand_1.row_count());
  _matrix_multiply_kernel->setArg(4, operand_2.row_count());
  _matrix_multiply_kernel->setArg(5, operand_2.column_count());

  cl::NDRange offset(0U, 0U);
  cl::NDRange workgroup_size(1U, 1U);
  cl::NDRange size(product.row_count(), product.column_count());

  _command_queue->enqueueNDRangeKernel(*_matrix_multiply_kernel, offset, size, workgroup_size);
  _command_queue->finish();
}

void opencl_context::multiply(matrix& product, const matrix& operand_1, const float operand_2) {
  opencl::matrix* product_impl = dynamic_cast<opencl::matrix*>(product.implementation());
  opencl::matrix* operand_1_impl = dynamic_cast<opencl::matrix*>(operand_1.implementation());

  _matrix_scalar_multiply->setArg(0, *product_impl->get());
  _matrix_scalar_multiply->setArg(1, *operand_1_impl->get());
  _matrix_scalar_multiply->setArg(2, operand_2);
  _matrix_scalar_multiply->setArg(3, product.row_count());
  _matrix_scalar_multiply->setArg(4, product.column_count());

  cl::NDRange offset(0U, 0U);
  cl::NDRange size(product.row_count(), product.column_count());
  _command_queue->enqueueNDRangeKernel(*_matrix_scalar_multiply, offset, size);
  _command_queue->finish();
}

void opencl_context::multiply_elementwise(matrix& product, const matrix& operand_1,
                                          const matrix& operand_2) {
  opencl::matrix* product_impl = dynamic_cast<opencl::matrix*>(product.implementation());
  opencl::matrix* operand_1_impl = dynamic_cast<opencl::matrix*>(operand_1.implementation());
  opencl::matrix* operand_2_impl = dynamic_cast<opencl::matrix*>(operand_2.implementation());

  _matrix_elementwise_multiply->setArg(0, *product_impl->get());
  _matrix_elementwise_multiply->setArg(1, *operand_1_impl->get());
  _matrix_elementwise_multiply->setArg(2, *operand_2_impl->get());
  _matrix_elementwise_multiply->setArg(3, product.row_count());
  _matrix_elementwise_multiply->setArg(4, product.column_count());

  cl::NDRange offset(0U, 0U);
  cl::NDRange workgroup_size(1U, 1U);
  cl::NDRange size(product.row_count(), product.column_count());

  _command_queue->enqueueNDRangeKernel(*_matrix_elementwise_multiply, offset, size, workgroup_size);
  _command_queue->finish();
}

void opencl_context::add(matrix& sum, const matrix& operand_1, const float operand_2) {
  opencl::matrix* sum_impl = dynamic_cast<opencl::matrix*>(sum.implementation());
  opencl::matrix* operand_1_impl = dynamic_cast<opencl::matrix*>(operand_1.implementation());

  _scalar_add->setArg(0, *sum_impl->get());
  _scalar_add->setArg(1, *operand_1_impl->get());
  _scalar_add->setArg(2, operand_2);
  _scalar_add->setArg(3, sum.row_count());
  _scalar_add->setArg(4, sum.column_count());

  cl::NDRange offset(0U, 0U);
  cl::NDRange workgroup_size(1U, 1U);
  cl::NDRange size(sum.row_count(), sum.column_count());

  _command_queue->enqueueNDRangeKernel(*_scalar_add, offset, size, workgroup_size);
  _command_queue->finish();
}

void opencl_context::add(matrix& sum, const matrix& operand_1, const matrix& operand_2) {
  opencl::matrix* sum_impl = dynamic_cast<opencl::matrix*>(sum.implementation());
  opencl::matrix* operand_1_impl = dynamic_cast<opencl::matrix*>(operand_1.implementation());
  opencl::matrix* operand_2_impl = dynamic_cast<opencl::matrix*>(operand_2.implementation());

  _matrix_add->setArg(0, *sum_impl->get());
  _matrix_add->setArg(1, *operand_1_impl->get());
  _matrix_add->setArg(2, *operand_2_impl->get());
  _matrix_add->setArg(3, sum.row_count());
  _matrix_add->setArg(4, sum.column_count());

  cl::NDRange offset(0U, 0U);
  cl::NDRange workgroup_size(1U, 1U);
  cl::NDRange size(sum.row_count(), sum.column_count());

  _command_queue->enqueueNDRangeKernel(*_matrix_add, offset, size, workgroup_size);
  _command_queue->finish();
}

void opencl_context::subtract(matrix& difference, const matrix& operand_1,
                              const matrix& operand_2) {
  opencl::matrix* difference_impl = dynamic_cast<opencl::matrix*>(difference.implementation());
  opencl::matrix* operand_1_impl = dynamic_cast<opencl::matrix*>(operand_1.implementation());
  opencl::matrix* operand_2_impl = dynamic_cast<opencl::matrix*>(operand_2.implementation());

  _matrix_subtract->setArg(0, *difference_impl->get());
  _matrix_subtract->setArg(1, *operand_1_impl->get());
  _matrix_subtract->setArg(2, *operand_2_impl->get());
  _matrix_subtract->setArg(3, difference.row_count());
  _matrix_subtract->setArg(4, difference.column_count());

  cl::NDRange offset(0U, 0U);
  cl::NDRange workgroup_size(1U, 1U);
  cl::NDRange size(difference.row_count(), difference.column_count());

  _command_queue->enqueueNDRangeKernel(*_matrix_subtract, offset, size, workgroup_size);
  _command_queue->finish();
}

void opencl_context::sigmoid(matrix& operand) {
  opencl::matrix* impl = (opencl::matrix*)operand.implementation();
  _matrix_sigmoid_kernel->setArg(0, *impl->get());
  _matrix_sigmoid_kernel->setArg(1, operand.row_count());
  _matrix_sigmoid_kernel->setArg(2, operand.column_count());

  cl::NDRange offset(0U, 0U);
  cl::NDRange size(operand.row_count(), operand.column_count());
  _command_queue->enqueueNDRangeKernel(*_matrix_sigmoid_kernel, offset, size);
  _command_queue->finish();
}

void opencl_context::sigmoid_gradient(matrix& gradient, const matrix& operand) {
  opencl::matrix* gradient_impl = dynamic_cast<opencl::matrix*>(gradient.implementation());
  opencl::matrix* operand_impl = dynamic_cast<opencl::matrix*>(operand.implementation());

  _sigmoid_gradient->setArg(0, *gradient_impl->get());
  _sigmoid_gradient->setArg(1, *operand_impl->get());
  _sigmoid_gradient->setArg(2, operand.row_count());
  _sigmoid_gradient->setArg(3, operand.column_count());

  cl::NDRange offset(0U, 0U);
  cl::NDRange size(operand.row_count(), operand.column_count());
  _command_queue->enqueueNDRangeKernel(*_sigmoid_gradient, offset, size);
  _command_queue->finish();
}

void opencl_context::hyperbolic_tangent(matrix& operand) {
  opencl::matrix* operand_impl = dynamic_cast<opencl::matrix*>(operand.implementation());

  _hyperbolic_tangent->setArg(0, *operand_impl->get());
  _hyperbolic_tangent->setArg(1, operand.row_count());
  _hyperbolic_tangent->setArg(2, operand.column_count());

  cl::NDRange offset(0U, 0U);
  cl::NDRange size(operand.row_count(), operand.column_count());
  _command_queue->enqueueNDRangeKernel(*_hyperbolic_tangent, offset, size);
  _command_queue->finish();
}

void opencl_context::hyperbolic_tangent_gradient(matrix& gradient, const matrix& operand) {
  opencl::matrix* gradient_impl = dynamic_cast<opencl::matrix*>(gradient.implementation());
  opencl::matrix* operand_impl = dynamic_cast<opencl::matrix*>(operand.implementation());

  _hyperbolic_tangent_gradient->setArg(0, *gradient_impl->get());
  _hyperbolic_tangent_gradient->setArg(1, *operand_impl->get());
  _hyperbolic_tangent_gradient->setArg(2, operand.row_count());
  _hyperbolic_tangent_gradient->setArg(3, operand.column_count());

  cl::NDRange offset(0U, 0U);
  cl::NDRange size(operand.row_count(), operand.column_count());
  _command_queue->enqueueNDRangeKernel(*_hyperbolic_tangent_gradient, offset, size);
  _command_queue->finish();
}

void opencl_context::softmax(matrix& operand) {
  opencl::matrix* impl = (opencl::matrix*)operand.implementation();
  _matrix_softmax_exp_kernel->setArg(0, *impl->get());
  _matrix_softmax_exp_kernel->setArg(1, operand.row_count());
  _matrix_softmax_exp_kernel->setArg(2, operand.column_count());

  cl::NDRange offset(0U, 0U);
  cl::NDRange exp_size(operand.row_count(), operand.column_count());
  _command_queue->enqueueNDRangeKernel(*_matrix_softmax_exp_kernel, offset, exp_size);

  _matrix_softmax_normalize_kernel->setArg(0U, *impl->get());
  _matrix_softmax_normalize_kernel->setArg(1U, operand.row_count());
  _matrix_softmax_normalize_kernel->setArg(2U, operand.column_count());

  cl::NDRange normalize_size(operand.row_count(), 1);
  _command_queue->enqueueNDRangeKernel(*_matrix_softmax_normalize_kernel, offset, normalize_size);

  _command_queue->finish();
}

void opencl_context::merge(matrix& merged, const matrix& operand_1, const matrix& operand_2) {
  opencl::matrix* merged_impl = dynamic_cast<opencl::matrix*>(merged.implementation());
  opencl::matrix* operand_1_impl = dynamic_cast<opencl::matrix*>(operand_1.implementation());
  opencl::matrix* operand_2_impl = dynamic_cast<opencl::matrix*>(operand_2.implementation());

  _matrix_merge_kernel->setArg(0U, *merged_impl->get());
  _matrix_merge_kernel->setArg(1U, *operand_1_impl->get());
  _matrix_merge_kernel->setArg(2U, *operand_2_impl->get());
  _matrix_merge_kernel->setArg(3U, merged.row_count());
  _matrix_merge_kernel->setArg(4U, operand_1.column_count());
  _matrix_merge_kernel->setArg(5U, operand_2.column_count());

  cl::NDRange offset(0U, 0U);
  cl::NDRange size(merged.row_count(), 1U);
  _command_queue->enqueueNDRangeKernel(*_matrix_merge_kernel, offset, size);
  _command_queue->finish();
}

void opencl_context::transpose(matrix& transposed, const matrix& original) {
  opencl::matrix* transposed_impl = dynamic_cast<opencl::matrix*>(transposed.implementation());
  opencl::matrix* original_impl = dynamic_cast<opencl::matrix*>(original.implementation());

  _matrix_transpose_kernel->setArg(0U, *transposed_impl->get());
  _matrix_transpose_kernel->setArg(1U, *original_impl->get());
  _matrix_transpose_kernel->setArg(2U, original.row_count());
  _matrix_transpose_kernel->setArg(3U, original.column_count());

  cl::NDRange offset(0U, 0U);
  cl::NDRange size(transposed.column_count(), transposed.row_count());
  _command_queue->enqueueNDRangeKernel(*_matrix_transpose_kernel, offset, size);
  _command_queue->finish();
}

matrix opencl_context::sum_rows(const matrix& original) {
  vi::la::matrix sums(*this, 1U, original.column_count(), 0.0);
  opencl::matrix* sum_impl = dynamic_cast<opencl::matrix*>(sums.implementation());
  opencl::matrix* original_imp = dynamic_cast<opencl::matrix*>(original.implementation());

  _sum_rows->setArg(0U, *sum_impl->get());
  _sum_rows->setArg(1U, *original_imp->get());
  _sum_rows->setArg(2U, original.row_count());
  _sum_rows->setArg(3U, original.column_count());

  cl::NDRange offset(0U, 0U);
  cl::NDRange size(1, sums.column_count());
  _command_queue->enqueueNDRangeKernel(*_sum_rows, offset, size);
  _command_queue->finish();
  return sums;
}

matrix opencl_context::sum_columns(const matrix& original) {
  vi::la::matrix sums(*this, original.row_count(), 1U, 0.0);
  opencl::matrix* sum_impl = dynamic_cast<opencl::matrix*>(sums.implementation());
  opencl::matrix* original_imp = dynamic_cast<opencl::matrix*>(original.implementation());

  _sum_columns->setArg(0U, *sum_impl->get());
  _sum_columns->setArg(1U, *original_imp->get());
  _sum_columns->setArg(2U, original.row_count());
  _sum_columns->setArg(3U, original.column_count());

  cl::NDRange offset(0U, 0U);
  cl::NDRange size(sums.row_count(), 1);
  _command_queue->enqueueNDRangeKernel(*_sum_columns, offset, size);
  _command_queue->finish();
  return sums;
}

void opencl_context::log(matrix& result, const matrix& original) {
  opencl::matrix* result_impl = dynamic_cast<opencl::matrix*>(result.implementation());
  opencl::matrix* original_imp = dynamic_cast<opencl::matrix*>(original.implementation());

  _log->setArg(0U, *result_impl->get());
  _log->setArg(1U, *original_imp->get());
  _log->setArg(2U, result.row_count());
  _log->setArg(3U, result.column_count());

  cl::NDRange offset(0U, 0U);
  cl::NDRange size(result.row_count(), result.column_count());
  _command_queue->enqueueNDRangeKernel(*_log, offset, size);
  _command_queue->finish();
}

void opencl_context::sub_matrix(matrix& target, const matrix& original, size_t start_row,
                                size_t end_row, size_t start_column, size_t end_column) {
  opencl::matrix* target_impl = dynamic_cast<opencl::matrix*>(target.implementation());
  opencl::matrix* original_imp = dynamic_cast<opencl::matrix*>(original.implementation());

  cl::size_t<3> source_origin;
  source_origin[0] = start_column * sizeof(float);
  source_origin[1] = start_row;
  source_origin[2] = 0;
  cl::size_t<3> destination_origin;
  destination_origin[0] = 0;
  destination_origin[1] = 0;
  destination_origin[2] = 0;

  const size_t sub_rows(end_row - start_row + 1U);
  const size_t sub_columns(end_column - start_column + 1U);

  cl::size_t<3> region;
  region[0] = sub_columns * sizeof(float);
  region[1] = sub_rows;
  region[2] = 1;

  cl_int error = _command_queue->enqueueCopyBufferRect(
      *original_imp->get(), *target_impl->get(), source_origin, destination_origin, region,
      original.column_count() * sizeof(float), 0U, target.column_count() * sizeof(float), 0U);
  assert(error == CL_SUCCESS);
  _command_queue->finish();
}

void opencl_context::convolve_2d(matrix& result, const matrix& mask, const matrix& original,
                                 size_t channels) {
  opencl::matrix* result_impl = dynamic_cast<opencl::matrix*>(result.implementation());
  opencl::matrix* mask_impl = dynamic_cast<opencl::matrix*>(mask.implementation());
  opencl::matrix* original_impl = dynamic_cast<opencl::matrix*>(original.implementation());

  // physical values
  const size_t OUTPUT_TILE_WIDTH = 4U * channels;
  const size_t OUTPUT_TILE_HEIGHT = 4U;

  // channels are interleaved
  const size_t MASK_RADIUS = mask.column_count() / 2;
  const size_t INPUT_TILE_WIDTH = OUTPUT_TILE_WIDTH + 2 * MASK_RADIUS * channels;
  const size_t INPUT_TILE_HEIGHT = OUTPUT_TILE_HEIGHT + 2 * MASK_RADIUS;

  // TODO: maximize input dims while ensuring:
  // items_per_group = (INPUT_TILE_WIDTH * INPUT_TILE_HEIGHT) <
  // CL_KERNEL_WORK_GROUP_SIZE
  //            auto devices = _context.getInfo<CL_CONTEXT_DEVICES>();
  //            for (auto & device : devices) {
  //                const cl_kernel_work_group_info workgroup_size =
  //                _convolve_2d->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
  //                std::cout << "workgroup size: " << workgroup_size <<
  //                std::endl;
  //            }
  _convolve_2d->setArg(0U, *result_impl->get());
  _convolve_2d->setArg(1U, *original_impl->get());
  _convolve_2d->setArg(2U, original.row_count());
  _convolve_2d->setArg(3U, original.column_count());
  _convolve_2d->setArg(4U, channels);
  _convolve_2d->setArg(5U, *mask_impl->get());
  _convolve_2d->setArg(6U, mask.row_count());
  _convolve_2d->setArg(7U, mask.column_count());
  _convolve_2d->setArg(8U, cl::__local(INPUT_TILE_HEIGHT * INPUT_TILE_WIDTH * sizeof(cl_float)));
  _convolve_2d->setArg(9U, OUTPUT_TILE_HEIGHT);
  _convolve_2d->setArg(10U, OUTPUT_TILE_WIDTH);
  const size_t data_width = original.column_count();
  const size_t data_height = original.row_count();

  const size_t horizontal_groups = ceil(((float)data_width) / OUTPUT_TILE_WIDTH);
  const size_t vertical_groups = ceil(((float)data_height) / OUTPUT_TILE_HEIGHT);

  cl::NDRange offset(0, 0);
  cl::NDRange items(vertical_groups * INPUT_TILE_HEIGHT, horizontal_groups * INPUT_TILE_WIDTH);
  cl::NDRange items_per_group(INPUT_TILE_HEIGHT, INPUT_TILE_WIDTH);

  _command_queue->enqueueNDRangeKernel(*_convolve_2d, offset, items, items_per_group);
  _command_queue->finish();
}
}
}
