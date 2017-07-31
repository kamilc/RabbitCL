#include <iostream>

#include "gtest/gtest.h"

#include <boost/numeric/ublas/matrix.hpp>

#include "input.h"
#include "dense.h"
#include "function/tanh.h"
#include "function/relu.h"
#include "matrix.h"

using namespace heed;

namespace ublas = boost::numeric::ublas;

TEST(sample_test_case, sample_test)
{
    auto root = input<float, mode::cpu>::define(10);
    auto hidden = dense<float, mode::cpu>::define(15, root, function::tanh<float, mode::cpu>());

    //auto hidden2 = std::make_shared<activation<float, function::tanh<float>>>(activation<float, function::tanh<float>>(hidden));
    //auto output = std::make_shared<

    auto network = hidden;

    EXPECT_EQ(15, network->size());

    // std::vector<float> data = { 1, 0, 1,
    //                             1, 1, 0,
    //                             0, 1, 1,
    //                             0, 1, 0 };

    // ublas::matrix<float> m(4, 3, 0);
    // std::copy(data.begin(), data.end(), m.data().begin());

    matrix<float> data = matrix<float>(mode::gpu, 4, 3, { 1, 0, 1,
                                                          1, 1, 0,
                                                          0, 1, 1,
                                                          0, 1, 0});
    matrix<float> ys = matrix<float>(mode::gpu, 4, 1, { 1, 1, 0, 0 });

    //network.train(data, ys, 100, 0.001);
}