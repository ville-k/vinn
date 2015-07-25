include_directories(${gtest_SOURCE_DIR}/include)

file(GLOB unit_test_SOURCES "*.cpp")
file(GLOB unit_test_HEADERS "*.h")
set(GLOBAL_SOURCES_TO_FORMAT
  ${GLOBAL_SOURCES_TO_FORMAT}
  ${unit_test_SOURCES}
  ${unit_test_HEADERS}
  PARENT_SCOPE
)


add_executable(unit_tests ${unit_test_SOURCES})
target_link_libraries(unit_tests ViNN_STATIC)
target_link_libraries(unit_tests gtest_main)
target_compile_definitions(unit_tests PRIVATE -DCL_USE_DEPRECATED_OPENCL_1_1_APIS=1)

add_test(AllUnitTestsInViNN unit_tests)

add_custom_target(coverage ALL tests/unit_tests
  COMMAND lcov -q --directory ${PROJECT_BINARY_DIR} --capture -o raw_coverage.info
  COMMAND lcov -q -r raw_coverage.info "*/gtest*" -o without_libs_coverage.info
  COMMAND lcov -q -r without_libs_coverage.info "*/cl.hpp" -o without_cl_hpp_coverage.info
  COMMAND lcov -q -r without_cl_hpp_coverage.info "*/include/*" -o coverage.info
  COMMAND genhtml -o coverage coverage.info
  COMMENT Generating code coverage report for unit tests
  DEPENDS unit_tests
  BYPRODUCTS coverage.info
  WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
)
set_target_properties(coverage PROPERTIES
  EXCLUDE_FROM_ALL TRUE
)

