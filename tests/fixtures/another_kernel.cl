
__kernel void subtract_numbers(__global float * a, float b) {
  float c = a[0] - b;
}

