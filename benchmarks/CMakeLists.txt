file(GLOB benchmark_SOURCES "*.cpp")
file(GLOB benchmark_HEADERS "*.h")
set(GLOBAL_SOURCES_TO_FORMAT
  ${GLOBAL_SOURCES_TO_FORMAT}
  ${benchmark_SOURCES}
  ${benchmark_HEADERS}
  PARENT_SCOPE
)

add_executable(benchmarks ${benchmark_SOURCES})
target_link_libraries(benchmarks ViNN_STATIC)
target_link_libraries(benchmarks benchmark)

