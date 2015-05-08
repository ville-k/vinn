set(vinn_GENERATED_OPENCL_SOURCES
  ${PROJECT_SOURCE_DIR}/src/vi/la/opencl/kernels_generated/generated_opencl_sources.h;
  ${PROJECT_SOURCE_DIR}/src/vi/la/opencl/kernels_generated/generated_opencl_sources.cpp;
  ${PROJECT_SOURCE_DIR}/src/vi/la/opencl/kernels_generated/activation_functions_cl__source.h;
  ${PROJECT_SOURCE_DIR}/src/vi/la/opencl/kernels_generated/activation_functions_cl__source.cpp;
  ${PROJECT_SOURCE_DIR}/src/vi/la/opencl/kernels_generated/convolution_cl__source.h;
  ${PROJECT_SOURCE_DIR}/src/vi/la/opencl/kernels_generated/convolution_cl__source.cpp;
  ${PROJECT_SOURCE_DIR}/src/vi/la/opencl/kernels_generated/matrix_cl__source.h;
  ${PROJECT_SOURCE_DIR}/src/vi/la/opencl/kernels_generated/matrix_cl__source.cpp
)

set(vinn_OPENCL_SOURCES
  ${PROJECT_SOURCE_DIR}/src/vi/la/opencl/kernels/activation_functions.cl;
  ${PROJECT_SOURCE_DIR}/src/vi/la/opencl/kernels/convolution.cl;
  ${PROJECT_SOURCE_DIR}/src/vi/la/opencl/kernels/matrix.cl
)

set(vinn_GENERATOR_ARGUMENTS
  ${PROJECT_SOURCE_DIR}/src/vi/la/opencl/kernels
  ${PROJECT_SOURCE_DIR}/src/vi/la/opencl/kernels_generated
  vi::la::opencl_generated
  vi::la::opencl::source
  vi/la/opencl/source.h
)

add_custom_command(
  OUTPUT ${vinn_GENERATED_OPENCL_SOURCES}
  COMMAND ${PROJECT_SOURCE_DIR}/bin/convert_opencl_source.rb ${vinn_GENERATOR_ARGUMENTS}
  DEPENDS ${vinn_OPENCL_SOURCES}
  COMMENT "Generating c++ files from OpenCL sources"
)


file(GLOB_RECURSE vinn_SOURCES *.cpp)
file(GLOB_RECURSE vinn_HEADERS *.h)
set(GLOBAL_SOURCES_TO_FORMAT
  ${GLOBAL_SOURCES_TO_FORMAT}
  ${vinn_SOURCES}
  ${vinn_HEADERS}
  PARENT_SCOPE
)

add_library(vinn_STATIC STATIC ${vinn_SOURCES} ${vinn_GENERATED_OPENCL_SOURCES})
set_target_properties(vinn_STATIC PROPERTIES
  OUTPUT_NAME vinn
  CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror"
)

add_library(vinn_SHARED SHARED ${vinn_SOURCES} ${vinn_GENERATED_OPENCL_SOURCES})
set_target_properties(vinn_SHARED PROPERTIES
  OUTPUT_NAME vinn
  CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror"
)

target_link_libraries(vinn_STATIC ${OpenCL_LIBRARIES})
target_link_libraries(vinn_SHARED ${OpenCL_LIBRARIES})

install(TARGETS vinn_STATIC DESTINATION lib)
install(TARGETS vinn_SHARED DESTINATION lib)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/src/"
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
  PATTERN "*/kernels_generated/*" EXCLUDE
)

