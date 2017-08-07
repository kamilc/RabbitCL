#include <iostream>

#include "gtest/gtest.h"

#include "sequence.h"
#include "input_config.h"
#include "dense_config.h"
#include "function/tanh.h"
#include "function/relu.h"
#include "function/softmax.h"
#include "gradient_descent.h"
#include "matrix_helpers.h"

using namespace mozart;
using namespace mozart::function;

TEST(relu_test_case, relu_test)
{
    auto data = make_matrix<float>({
        {-5, -4, -2 },
        { 0,  1, -1 },
        { 2,  1, -10},
    });

    auto result = relu<float>(data, false);

    EXPECT_EQ(result.out(0, 0), 0);
    EXPECT_EQ(result.out(0, 1), 0);
    EXPECT_EQ(result.out(0, 2), 0);
    EXPECT_EQ(result.out(1, 0), 0);
    EXPECT_EQ(result.out(1, 1), 1);
    EXPECT_EQ(result.out(1, 2), 0);
    EXPECT_EQ(result.out(2, 0), 2);
    EXPECT_EQ(result.out(2, 1), 1);
    EXPECT_EQ(result.out(2, 2), 0);
}

TEST(learn_binary_test_case, learn_binary_test)
{
    sequence<float> network;

    network.add(input_config<float>(3))
           .add(dense_config<float>(3, relu<float>))
           .add(dense_config<float>(8, softmax<float>));

    EXPECT_EQ(network.size(), 3);

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