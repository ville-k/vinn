#include <cstring>
#include "convolution_cl__source.h"
namespace vi {
namespace la {
namespace opencl_generated {
void convolution_cl__source(const char** name, const char** data, size_t& length) {
  *name = "convolution.cl";
  *data = "#if defined(DOUBLE_SUPPORT_AVAILABLE)\n#pragma OPENCL EXTENSION cl_khr_fp64 : "
          "enable\ntypedef double real_t;\n#else\ntypedef float real_t;\n#endif\n\n__kernel void "
          "matrix_convolve_2d(__global real_t * result, __global real_t * source,\n                "
          "                 size_t data_rows, size_t data_columns, size_t data_channels,\n         "
          "                        __constant real_t * mask, size_t mask_rows, size_t "
          "mask_columns,\n                                 __local real_t * input_tile, size_t "
          "output_rows, size_t output_columns) {\n\n  const size_t MASK_WIDTH = mask_columns;\n  "
          "const size_t MASK_RADIUS = (mask_columns/2);\n\n  const size_t OUTPUT_TILE_WIDTH = "
          "output_columns;\n  const size_t OUTPUT_TILE_HEIGHT = output_rows;\n\n  const size_t "
          "tile_columns = get_local_size(1);\n  const size_t tile_rows    = get_local_size(0);\n\n "
          " size_t output_row    = get_group_id(0) * OUTPUT_TILE_HEIGHT + get_local_id(0);\n  "
          "size_t output_column = get_group_id(1) * OUTPUT_TILE_WIDTH + get_local_id(1);\n\n  int "
          "input_row    = output_row    - MASK_RADIUS;\n  int input_column = output_column - "
          "MASK_RADIUS * data_channels;\n\n  if (input_row >= 0 && input_row < data_rows &&\n      "
          "input_column >= 0 && input_column < data_columns) {\n    input_tile[get_local_id(0) * "
          "tile_columns + get_local_id(1)] = source[input_row * data_columns + input_column];\n  } "
          "else {\n    input_tile[get_local_id(0) * tile_columns + get_local_id(1)] = 0.0;\n  "
          "}\n\n  barrier(CLK_LOCAL_MEM_FENCE);\n#if 0\n  if (0 == get_group_id(0) && 0 == "
          "get_group_id(1) && 0 == get_local_id(0) && 0 == get_local_id(1)) {\n    printf(\"tile "
          "cols: %d, rows: %d\n\", tile_columns, tile_rows);\n    printf(\"output_col: %d, "
          "output_row: %d\n\", output_column, output_row);\n    printf(\"MASK_WIDTH: %d, "
          "MASK_RADIUS: %d\n\", MASK_WIDTH, MASK_RADIUS);\n\n    printf(\"INPUT TILE:\n\");\n    "
          "for (size_t m = 0U; m < tile_rows; ++m) {\n      for (size_t n = 0U; n < tile_columns; "
          "++n) {\n        printf(\"%f \", input_tile[m * tile_columns + n]);\n      }\n      "
          "printf(\"\n\");\n    }\n    printf(\"MASK\n\");\n      for (size_t m = 0U; m < "
          "MASK_WIDTH; ++m) {\n          for (size_t n = 0U; n < MASK_WIDTH; ++n) {\n              "
          "printf(\"%f \", mask[m * MASK_WIDTH + n]);\n          }\n          printf(\"\n\");\n    "
          "  }\n  }\n#endif\n\n  real_t output = 0.0;\n  if (get_local_id(0) < OUTPUT_TILE_HEIGHT "
          "&& get_local_id(1) < OUTPUT_TILE_WIDTH) {\n    for (size_t m = 0U; m < MASK_WIDTH; ++m) "
          "{\n      for (size_t n = 0U; n < MASK_WIDTH; ++n) {\n        size_t column_index = "
          "get_local_id(1) + n * data_channels;\n        output += mask[m * MASK_WIDTH + n] *\n    "
          "    input_tile[(get_local_id(0) + m) * tile_columns + column_index];\n      }\n    }\n  "
          "}\n\n  if (output_row < data_rows && output_column < data_columns) {\n    "
          "result[output_row * data_columns + output_column] = output;\n  }\n}\n\n";
  length = std::strlen(*data) + 1U;
  return;
}
}
}
}
