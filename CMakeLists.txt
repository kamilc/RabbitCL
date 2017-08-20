cmake_minimum_required(VERSION 2.8)

project(librabbit)

set (CMAKE_CXX_STANDARD 14)
add_compile_options(-std=c++14)
add_definitions(-DVIENNACL_WITH_OPENCL)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package( OpenCL )

enable_testing()

include(ExternalProject)
ExternalProject_Add(googletest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           master
  SOURCE_DIR        "${CMAKE_BINARY_DIR}/googletest-src"
  BINARY_DIR        "${CMAKE_BINARY_DIR}/googletest-build"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)

add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
                 ${CMAKE_BINARY_DIR}/googletest-build)

include(BuildBoostCompute)

include_directories(include /opt/local/include ${OPENCL_INCLUDE_DIR} ${BoostCompute_INCLUDE_DIRS})

add_library(mozart STATIC src/sequence.cpp
                          src/matrix.cpp
                          src/matrix_helpers.cpp
                          src/layer_config.cpp
                          src/layer.cpp
                          src/input_config.cpp
                          src/input.cpp
                          src/dense_config.cpp
                          src/dense.cpp
                          src/activation.cpp
                          src/cost.cpp
                          src/function/squared_error.cpp
                          src/function/reduce_avg.cpp
                          src/function/tanh.cpp
                          src/function/relu.cpp
                          src/function/softmax.cpp
                          src/random_matrix_generator.cpp
                          src/gradient_descent.cpp)

add_executable(mozart_test test/mozart_test.cpp)

message(status "Found OpenCL includes: ${OPENCL_INCLUDE_DIR}")
message(status "Found OpenCL libs: ${OPENCL_LIBRARIES}")

target_link_libraries(mozart_test mozart gtest_main pthread ${OPENCL_LIBRARIES})

add_test(NAME mozart_test COMMAND mozart_test)

install(TARGETS mozart DESTINATION lib)

install(FILES mozart.h DESTINATION include)