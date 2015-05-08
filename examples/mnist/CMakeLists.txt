file(GLOB vinn_example_mnist_SOURCES *.cpp)
set(GLOBAL_SOURCES_TO_FORMAT
  ${GLOBAL_SOURCES_TO_FORMAT}
  ${vinn_example_mnist_SOURCES}
  PARENT_SCOPE
)

add_executable(vinn_example_mnist ${vinn_example_mnist_SOURCES})
target_link_libraries(vinn_example_mnist vinn_STATIC)

