#include <iostream>

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION 1

#include "gtest/gtest.h"


#include "sequence.h"
#include "input_config.h"
#include "dense_config.h"
#include "function/tanh.h"
#include "function/relu.h"
#include "function/softmax.h"
#include "function/squared_error.h"
#include "gradient_descent.h"
#include "matrix_helpers.h"
#include "function/reduce_avg.h"

using namespace mozart;
using namespace mozart::function;

TEST(reduce_avg_test_case, reduce_avg_test)
{
    std::cout << "Still alive" << std::endl;
    auto data = make_matrix<float>({
        { 1, 2, 3.0 },
        { 6, 7, 9.5 }
    });
    std::cout << "Still alive 2" << std::endl;

    float result = reduce_avg<float>(data);

    EXPECT_NEAR(result, 4.75, 0.0001);
}

TEST(squared_error_test_case, squared_error_test)
{
    auto predicted = make_matrix<float>({
        { 0,  1, -1 },
        { 2,  1, -10},
    });

    auto targets = make_matrix<float>({
        { 1.5,   -2,   0 },
        { 2.1,  1.1, -10 }
    });

    auto result = squared_error<float>(predicted, targets, false);

    EXPECT_NEAR(result.out, -0.15, 0.0001);
}

TEST(squared_error_test_case, squared_error_deriv_test)
{
    auto predicted = make_matrix<float>({
        { 0,  1, -1 },
        { 2,  1, -10},
    });

    auto targets = make_matrix<float>({
        { 1.5,   -2,   0 },
        { 2.1,  1.1, -10 }
    });

    auto result = squared_error<float>(predicted, targets, false);

    EXPECT_NEAR(result.deriv(0, 0), 0 - 1.5, 0.0001);
    EXPECT_NEAR(result.deriv(0, 1), 1 + 2, 0.0001);
    EXPECT_NEAR(result.deriv(0, 2), -1 - 0, 0.0001);
    EXPECT_NEAR(result.deriv(1, 0), 2 - 2.1, 0.0001);
    EXPECT_NEAR(result.deriv(1, 1), 1 - 1.1, 0.0001);
    EXPECT_NEAR(result.deriv(1, 2), -10 + 10, 0.0001);
}

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

TEST(relu_test_case, relu_deriv_test)
{
    auto data = make_matrix<float>({
        {-5, -4, -2 },
        { 0,  1, -1 },
        { 2,  1, -10},
    });

    auto result = relu<float>(data, true);

    EXPECT_EQ(result.deriv(0, 0), 0);
    EXPECT_EQ(result.deriv(0, 1), 0);
    EXPECT_EQ(result.deriv(0, 2), 0);
    EXPECT_EQ(result.deriv(1, 0), 0);
    EXPECT_EQ(result.deriv(1, 1), 1);
    EXPECT_EQ(result.deriv(1, 2), 0);
    EXPECT_EQ(result.deriv(2, 0), 1);
    EXPECT_EQ(result.deriv(2, 1), 1);
    EXPECT_EQ(result.deriv(2, 2), 0);
}

TEST(tanh_test_case, tanh_test)
{
    auto data = make_matrix<float>({
        {-5, -4, -2 },
        { 0,  1, -1 },
        { 2,  1, -10},
    });

    auto result = tanh<float>(data, false);

    EXPECT_NEAR(result.out(0, 0), -0.9999092, 0.0001);
    EXPECT_NEAR(result.out(0, 1), -0.9993293, 0.0001);
    EXPECT_NEAR(result.out(0, 2), -0.96402758, 0.0001);
    EXPECT_NEAR(result.out(1, 0), 0, 0.0001);
    EXPECT_NEAR(result.out(1, 1), 0.76159416, 0.0001);
    EXPECT_NEAR(result.out(1, 2), -0.76159416, 0.0001);
    EXPECT_NEAR(result.out(2, 0), 0.96402758, 0.0001);
    EXPECT_NEAR(result.out(2, 1), 0.76159416, 0.0001);
    EXPECT_NEAR(result.out(2, 2), -1, 0.0001);
}

TEST(tanh_test_case, tanh_deriv_test)
{
    auto data = make_matrix<float>({
        {-5, -4, -2 },
        { 0,  1, -1 },
        { 2,  1, -10},
    });

    auto result = tanh<float>(data, true);

    EXPECT_NEAR(result.deriv(0, 0), 0.00018158, 0.0001);
    EXPECT_NEAR(result.deriv(0, 1), 0.00134095, 0.0001);
    EXPECT_NEAR(result.deriv(0, 2), 0.07065082, 0.0001);
    EXPECT_NEAR(result.deriv(1, 0), 1, 0.0001);
    EXPECT_NEAR(result.deriv(1, 1), 0.41997434, 0.0001);
    EXPECT_NEAR(result.deriv(1, 2), 0.41997434, 0.0001);
    EXPECT_NEAR(result.deriv(2, 0), 7.06508249e-02, 0.0001);
    EXPECT_NEAR(result.deriv(2, 1), 4.19974342e-01, 0.0001);
    EXPECT_NEAR(result.deriv(2, 2), 8.24461455e-09, 0.0001);
}

TEST(softmax_test_case, softmax_test)
{
    auto data = make_matrix<float>({
        {  3.0, 1.0,  0.2 },
        {  3.0, 1.5, -0.2 },
        { -3.0, 1.5, -0.2 },
    });

    auto result = softmax<float>(data, false);

    viennacl::backend::finish();

    EXPECT_NEAR(result.out(0, 0), 0.8360188, 0.0001);
    EXPECT_NEAR(result.out(0, 1), 0.11314284, 0.0001);
    EXPECT_NEAR(result.out(0, 2), 0.05083836, 0.0001);
    EXPECT_NEAR(result.out(1, 0), 0.79120662, 0.0001);
    EXPECT_NEAR(result.out(1, 1), 0.17654206, 0.0001);
    EXPECT_NEAR(result.out(1, 2), 0.03225133, 0.0001);
    EXPECT_NEAR(result.out(2, 0), 0.00930563, 0.0001);
    EXPECT_NEAR(result.out(2, 1), 0.8376665, 0.0001);
    EXPECT_NEAR(result.out(2, 2), 0.15302787, 0.0001);
}

TEST(softmax_test_case, softmax_deriv_test)
{
    auto data = make_matrix<float>({
        {  3.0, 1.0,  0.2 },
        {  3.0, 1.5, -0.2 },
        { -3.0, 1.5, -0.2 },
    });

    auto result = softmax<float>(data, true);

    EXPECT_NEAR(result.deriv(0, 0), 0.13709136, 0.0001);
    EXPECT_NEAR(result.deriv(0, 1), 0.10034154, 0.0001);
    EXPECT_NEAR(result.deriv(0, 2), 0.04825382, 0.0001);
    EXPECT_NEAR(result.deriv(1, 0), 0.16519871, 0.0001);
    EXPECT_NEAR(result.deriv(1, 1), 0.14537496, 0.0001);
    EXPECT_NEAR(result.deriv(1, 2), 0.03121118, 0.0001);
    EXPECT_NEAR(result.deriv(2, 0), 0.00921904, 0.0001);
    EXPECT_NEAR(result.deriv(2, 1), 0.13598134, 0.0001);
    EXPECT_NEAR(result.deriv(2, 2), 0.12961034, 0.0001);
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

    auto optimizer = gradient_descent<float>(squared_error<float>)
                        .setEpochs(8*10)
                        .setBatches(1);

    optimizer.run(network, data, ys);

    auto test = make_matrix<float>({{ 0, 1, 0 }});
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