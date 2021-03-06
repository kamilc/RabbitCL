cmake_minimum_required(VERSION 2.8)

project(librabbit)

set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

set (CMAKE_CXX_STANDARD 14)
add_compile_options(-std=c++14)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package( OpenCL )

enable_testing()

include(BuildBoostCompute)
include(BuildClBLASt)
include(BuildRang)

#message("Found Boost Compute includes: ${BoostCompute_INCLUDE_DIRS}")

include_directories(include
                    /opt/local/include
                    ${OPENCL_INCLUDE_DIR}
                    ${BoostCompute_INCLUDE_DIRS}
                    ${ClBlast_INCLUDE}
                    ${Rang_INCLUDE}
                    )

add_library(mozart STATIC src/sequence.cpp
                          src/matrix.cpp
                          src/scalar.cpp
                          src/matrix_helpers.cpp
                          src/layer_config.cpp
                          src/layer.cpp
                          src/input_config.cpp
                          src/input.cpp
                          src/dense_config.cpp
                          src/dense.cpp
                          src/activation.cpp
                          src/cost.cpp
                          src/stat.cpp
                          src/observer.cpp
                          src/observer/timed.cpp
                          src/stats/accuracy.cpp
                          src/opencl/dot.cpp
                          src/opencl/scale.cpp
                          src/opencl/scalar_translate.cpp
                          src/opencl/element_mul.cpp
                          src/opencl/element_add.cpp
                          src/opencl/element_add_assign.cpp
                          src/opencl/squared_error.cpp
                          src/opencl/categorical_cross_entropy.cpp
                          src/opencl/reduce_avg.cpp
                          src/opencl/tanh.cpp
                          src/opencl/relu.cpp
                          src/opencl/softmax.cpp
                          src/opencl/squashmax.cpp
                          src/opencl/inplace_columnwise_subtract.cpp
                          src/opencl/inplace_reduce_column_sum.cpp
                          src/opencl/adagrad_update.cpp
                          src/opencl/rmsprop_update.cpp
                          src/opencl/accuracy_rate.cpp
                          src/random_matrix_generator.cpp
                          src/optimizers/gradient_descent.cpp
                          src/optimizers/adagrad.cpp
                          src/optimizers/rmsprop.cpp
                          )

message("Found OpenCL includes: ${OPENCL_INCLUDE_DIR}")
message("Found OpenCL libs: ${OPENCL_LIBRARIES}")

message("ClBlast includes: ${ClBlast_INCLUDE}")
message("ClBlast lib: ${ClBlast_LIB}")

add_executable(simple_feed_forward_example examples/simple_feed_forward_example.cpp)
target_link_libraries(simple_feed_forward_example mozart pthread ${OPENCL_LIBRARIES} ${ClBlast_LIB})

add_executable(adagrad_feed_forward_example examples/adagrad_feed_forward_example.cpp)
target_link_libraries(adagrad_feed_forward_example mozart pthread ${OPENCL_LIBRARIES} ${ClBlast_LIB})

add_executable(rmsprop_feed_forward_example examples/rmsprop_feed_forward_example.cpp)
target_link_libraries(rmsprop_feed_forward_example mozart pthread ${OPENCL_LIBRARIES} ${ClBlast_LIB})

add_executable(rmsprop_big_matrices_example examples/rmsprop_big_matrices_example.cpp)
target_link_libraries(rmsprop_big_matrices_example mozart pthread ${OPENCL_LIBRARIES} ${ClBlast_LIB})

add_executable(xor_example examples/xor_example.cpp)
target_link_libraries(xor_example mozart pthread ${OPENCL_LIBRARIES} ${ClBlast_LIB})

add_executable(xor_adagrad_example examples/xor_adagrad_example.cpp)
target_link_libraries(xor_adagrad_example mozart pthread ${OPENCL_LIBRARIES} ${ClBlast_LIB})

add_executable(xor_rmsprop_example examples/xor_rmsprop_example.cpp)
target_link_libraries(xor_rmsprop_example mozart pthread ${OPENCL_LIBRARIES} ${ClBlast_LIB})

add_executable(mozart_test test/mozart_test.cpp)
target_link_libraries(mozart_test mozart /opt/local/lib/libgtest_main.a /opt/local/lib/libgtest.a pthread ${OPENCL_LIBRARIES} ${ClBlast_LIB})

# todo: add the clblast resolution

add_test(NAME mozart_test COMMAND mozart_test)

install(TARGETS mozart DESTINATION lib)

install(FILES mozart.h DESTINATION include)
