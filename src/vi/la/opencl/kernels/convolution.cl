#if defined(DOUBLE_SUPPORT_AVAILABLE)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real_t;
#else
typedef float real_t;
#endif

__kernel void matrix_convolve_2d(__global real_t * result, __global real_t * source,
                                 size_t data_rows, size_t data_columns, size_t data_channels,
                                 __constant real_t * mask, size_t mask_rows, size_t mask_columns,
                                 __local real_t * input_tile, size_t output_rows, size_t output_columns) {

  const size_t MASK_WIDTH = mask_columns;
  const size_t MASK_RADIUS = (mask_columns/2);

  const size_t OUTPUT_TILE_WIDTH = output_columns;
  const size_t OUTPUT_TILE_HEIGHT = output_rows;

  const size_t tile_columns = get_local_size(1);
  const size_t tile_rows    = get_local_size(0);

  size_t output_row    = get_group_id(0) * OUTPUT_TILE_HEIGHT + get_local_id(0);
  size_t output_column = get_group_id(1) * OUTPUT_TILE_WIDTH + get_local_id(1);

  int input_row    = output_row    - MASK_RADIUS;
  int input_column = output_column - MASK_RADIUS * data_channels;

  if (input_row >= 0 && input_row < data_rows &&
      input_column >= 0 && input_column < data_columns) {
    input_tile[get_local_id(0) * tile_columns + get_local_id(1)] = source[input_row * data_columns + input_column];
  } else {
    input_tile[get_local_id(0) * tile_columns + get_local_id(1)] = 0.0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
#if 0
  if (0 == get_group_id(0) && 0 == get_group_id(1) && 0 == get_local_id(0) && 0 == get_local_id(1)) {
    printf("tile cols: %d, rows: %d\n", tile_columns, tile_rows);
    printf("output_col: %d, output_row: %d\n", output_column, output_row);
    printf("MASK_WIDTH: %d, MASK_RADIUS: %d\n", MASK_WIDTH, MASK_RADIUS);

    printf("INPUT TILE:\n");
    for (size_t m = 0U; m < tile_rows; ++m) {
      for (size_t n = 0U; n < tile_columns; ++n) {
        printf("%f ", input_tile[m * tile_columns + n]);
      }
      printf("\n");
    }
    printf("MASK\n");
      for (size_t m = 0U; m < MASK_WIDTH; ++m) {
          for (size_t n = 0U; n < MASK_WIDTH; ++n) {
              printf("%f ", mask[m * MASK_WIDTH + n]);
          }
          printf("\n");
      }
  }
#endif

  real_t output = 0.0;
  if (get_local_id(0) < OUTPUT_TILE_HEIGHT && get_local_id(1) < OUTPUT_TILE_WIDTH) {
    for (size_t m = 0U; m < MASK_WIDTH; ++m) {
      for (size_t n = 0U; n < MASK_WIDTH; ++n) {
        size_t column_index = get_local_id(1) + n * data_channels;
        output += mask[m * MASK_WIDTH + n] *
        input_tile[(get_local_id(0) + m) * tile_columns + column_index];
      }
    }
  }

  if (output_row < data_rows && output_column < data_columns) {
    result[output_row * data_columns + output_column] = output;
  }
}

