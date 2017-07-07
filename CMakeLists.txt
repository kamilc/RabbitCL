cmake_minimum_required(VERSION 2.8)

project(librabbit)

set (CMAKE_CXX_STANDARD 11)
add_compile_options(-std=c++11)

find_package( OpenCL REQUIRED )

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

include_directories(include /opt/local/include ${OPENCL_INCLUDE_DIR})

add_library(rabbit STATIC src/Layer.cpp
                          src/Activation.cpp
                          src/TanhActivation.cpp
                          src/IdentityActivation.cpp
                          src/TrainDef.cpp
                          src/Distributions.cpp
                          src/Memory.cpp)

add_executable(rabbit_test test/rabbit_test.cpp)

target_link_libraries(rabbit_test rabbit gtest_main pthread ${OPENCL_LIBRARIES})

add_test(NAME rabbit_test COMMAND rabbit_test)

install(TARGETS rabbit DESTINATION lib)

install(FILES rabbit.h DESTINATION include)