#include <iostream>

#include "gtest/gtest.h"

#include "sequence.h"
#include "input.h"
#include "dense.h"
#include "function/tanh.h"
#include "function/relu.h"
#include "function/softmax.h"
#include "gradient_descent.h"

using namespace heed;
using namespace heed::function;

TEST(sample_test_case, sample_test)
{
    sequence<float> network;

    network.add(input<float>::with(3))
           .add(dense<float>::with(3, relu<float>))
           .add(dense<float>::with(8, softmax<float>));

    // auto data = matrix<float, mode::cpu>(8, 3, { 0, 0, 0, /* 0 */
    //                                              0, 0, 1, /* 1 */
    //                                              0, 1, 0, /* 2 */
    //                                              0, 1, 1, /* 3 */
    //                                              1, 0, 0, /* 4 */
    //                                              1, 0, 1, /* 5 */
    //                                              1, 1, 0, /* 6 */
    //                                              1, 1, 1  /* 7 */ } );
    // auto ys = matrix<float, mode::cpu>(8, 8, { 1, 0, 0, 0, 0, 0, 0, 0, /* 0 */
    //                                            0, 1, 0, 0, 0, 0, 0, 0, /* 1 */
    //                                            0, 0, 1, 0, 0, 0, 0, 0, /* 2 */
    //                                            0, 0, 0, 1, 0, 0, 0, 0, /* 3 */
    //                                            0, 0, 0, 0, 1, 0, 0, 0, /* 4 */
    //                                            0, 0, 0, 0, 0, 1, 0, 0, /* 5 */
    //                                            0, 0, 0, 0, 0, 0, 1, 0, /* 6 */
    //                                            0, 0, 0, 0, 0, 0, 0, 1 /* 7 */
    //                                          });

    auto optimizer = gradient_descent<float>()
                        .setEpochs(8*10)
                        .setBatches(1);
    
    // optimizer.run(network, data, ys);

    // auto test = matrix<float, mode::cpu>(1, 3, { 0, 1, 0 });
    // auto expecting = matrix<float, mode::cpu>(1, 8, { 0, 0, 1, 0, 0, 0, 0, 0 });
    // auto predicted = network.forward(test);

    // EXPECT_EQ(predicted == expecting, true);
}