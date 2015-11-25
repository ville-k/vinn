cmake_minimum_required(VERSION 3.2)
project(vinn)

set(vinn_VERSION_MAJOR 0)
set(vinn_VERSION_MINOR 2)
set(vinn_VERSION_PATCH 0)
set(vinn_VERSION "${vinn_VERSION_MAJOR}.${vinn_VERSION_MINOR}.${vinn_VERSION_PATCH}")


find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not specified, defaulting to Debug")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

add_definitions(-DCONFIGURATION=${CMAKE_BUILD_TYPE})
add_definitions(-DSRCROOT="${PROJECT_SOURCE_DIR}")
add_definitions(-D__CL_ENABLE_EXCEPTIONS=1)
add_definitions(-std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  # Treat the missing c++ wrapper as a system include to avoid warnings
  include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/src/vi/la/opencl/darwin")
else()
  # gcc 4.8 has TR1 tuple
  add_definitions(-DGTEST_USE_OWN_TR1_TUPLE=0)
endif()


set(COVERAGE_COMPILE_FLAGS "-fprofile-arcs -ftest-coverage")
set(COVERAGE_LINK_FLAGS "-fprofile-arcs -ftest-coverage")
# Debug builds always produce coverage information
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${COVERAGE_COMPILE_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} ${COVERAGE_LINK_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} ${COVERAGE_LINK_FLAGS}")

include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/src")


# Individual targets will append source files for the format target
set(GLOBAL_SOURCES_TO_FORMAT "")
enable_testing()
add_subdirectory(src)
add_subdirectory(lib/googletest)
add_subdirectory(lib/benchmark)
add_subdirectory(bindings/python)
add_subdirectory(tests)
add_subdirectory(benchmarks)

add_custom_target(format
  COMMENT Format source code using clang-format
)

foreach(source_file ${GLOBAL_SOURCES_TO_FORMAT})
  add_custom_command(TARGET format POST_BUILD
    COMMAND clang-format -style=file -i ${source_file}
  )
endforeach(source_file)

set_target_properties(format PROPERTIES
  EXCLUDE_FROM_ALL TRUE
)

include(InstallRequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR "${vinn_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${vinn_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${vinn_VERSION_PATCH}")
include(CPack)

