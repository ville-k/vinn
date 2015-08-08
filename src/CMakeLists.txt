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

add_library(ViNN_STATIC STATIC 
  ${vinn_SOURCES} 
  ${vinn_GENERATED_OPENCL_SOURCES}
)
set_target_properties(ViNN_STATIC
  PROPERTIES
  OUTPUT_NAME vinn
  CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror"
)
target_compile_options(ViNN_STATIC
  PUBLIC -std=c++11
)

add_library(ViNN SHARED
  ${vinn_SOURCES}
  ${vinn_GENERATED_OPENCL_SOURCES}
)
set_target_properties(ViNN
  PROPERTIES
  OUTPUT_NAME vinn
  CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror"
)
target_compile_options(ViNN
  PUBLIC -std=c++11
)

set(LIBRARIES
  ${OpenCL_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
)
target_link_libraries(ViNN_STATIC PUBLIC ${LIBRARIES})
target_compile_definitions(ViNN_STATIC PRIVATE -DCL_USE_DEPRECATED_OPENCL_1_1_APIS=1)
target_link_libraries(ViNN PUBLIC ${LIBRARIES})
target_compile_definitions(ViNN PRIVATE -DCL_USE_DEPRECATED_OPENCL_1_1_APIS=1)

set(INCLUDE_INSTALL_DIR include CACHE PATH "Header installation directory")
set(LIBRARY_INSTALL_DIR lib CACHE PATH "Library installation directory")


install(TARGETS ViNN_STATIC DESTINATION ${LIBRARY_INSTALL_DIR})
install(TARGETS ViNN DESTINATION ${LIBRARY_INSTALL_DIR})
install(DIRECTORY "${PROJECT_SOURCE_DIR}/src/"
  DESTINATION ${INCLUDE_INSTALL_DIR}
  FILES_MATCHING PATTERN "*.h"
  PATTERN "*/kernels_generated/*" EXCLUDE
)


include(CMakePackageConfigHelpers)
set(ConfigPackageLocation lib/cmake/ViNN)

# Target export set
install(TARGETS ViNN ViNN_STATIC
  EXPORT ViNNTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(EXPORT ViNNTargets
  FILE ViNNTargets.cmake
  NAMESPACE ViNN::
  DESTINATION ${ConfigPackageLocation}
)

configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/ViNNConfig.cmake.in 
  "${CMAKE_CURRENT_BINARY_DIR}/ViNN/ViNNConfig.cmake"
  INSTALL_DESTINATION ${LIBRARY_INSTALL_DIR}/cmake/ViNN
  PATH_VARS INCLUDE_INSTALL_DIR LIBRARY_INSTALL_DIR LIBRARIES
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/ViNN/ViNNConfigVersion.cmake"
  VERSION ${vinn_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/ViNN/ViNNConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/ViNN/ViNNConfigVersion.cmake
  DESTINATION ${ConfigPackageLocation}
  COMPONENT Development
)

