#include <iostream>

#include "gtest/gtest.h"

#include "sequence.h"
#include "input.h"
#include "dense.h"
#include "function/tanh.h"
#include "function/relu.h"
#include "function/softmax.h"
#include "gradient_descent.h"
#include "matrix_helpers.h"

using namespace mozart;
using namespace mozart::function;

TEST(sample_test_case, sample_test)
{
    sequence<float> network;

    network.add(input<float>::with(3))
           .add(dense<float>::with(3, relu<float>))
           .add(dense<float>::with(8, softmax<float>));

    auto data = make_matrix<float>({
        {0, 0, 0},
        {0, 0, 1}, /* 1 */
        {0, 1, 0}, /* 2 */
        {0, 1, 1}, /* 3 */
        {1, 0, 0}, /* 4 */
        {1, 0, 1}, /* 5 */
        {1, 1, 0}, /* 6 */
        {1, 1, 1}  /* 7 */
    });

     auto ys = make_matrix<float>({
        {1, 0, 0, 0, 0, 0, 0, 0}, /* 0 */
        {0, 1, 0, 0, 0, 0, 0, 0}, /* 1 */
        {0, 0, 1, 0, 0, 0, 0, 0}, /* 2 */
        {0, 0, 0, 1, 0, 0, 0, 0}, /* 3 */
        {0, 0, 0, 0, 1, 0, 0, 0}, /* 4 */
        {0, 0, 0, 0, 0, 1, 0, 0}, /* 5 */
        {0, 0, 0, 0, 0, 0, 1, 0}, /* 6 */
        {0, 0, 0, 0, 0, 0, 0, 1} /* 7 */                    
     });

    auto optimizer = gradient_descent<float>()
                        .setEpochs(8*10)
                        .setBatches(1);

    optimizer.run(network, data, ys);

    auto test = make_matrix<float>({{ 0, 1, 0 }});
    auto expecting = make_matrix<float>({{ 0, 0, 1, 0, 0, 0, 0, 0 }});
    auto predicted = network.forward(test);

    EXPECT_EQ(predicted(0, 0), 0);
    EXPECT_EQ(predicted(0, 1), 0);
    EXPECT_EQ(predicted(0, 2), 1);
    EXPECT_EQ(predicted(0, 3), 0);
    EXPECT_EQ(predicted(0, 4), 0);
    EXPECT_EQ(predicted(0, 5), 0);
    EXPECT_EQ(predicted(0, 6), 0);
    EXPECT_EQ(predicted(0, 7), 0);
}