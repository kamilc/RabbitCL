#include <iostream>

#include "gtest/gtest.h"

#include "input.h"
#include "dense.h"
#include "activation.h"
#include "function/tanh.h"
#include "function/relu.h"

using namespace heed;

TEST(sample_test_case, sample_test)
{
    auto root = std::make_shared<input<float>>(input<float>(10));
    auto hidden = std::make_shared<dense<float>>(dense<float>(15, root));
    auto network = std::make_shared<activation<float, function::tanh<float>>>(activation<float, function::tanh<float>>(hidden));
    auto network2 = std::make_shared<activation<float, function::relu<float>>>(activation<float, function::relu<float>>(hidden));

    EXPECT_EQ(15, network->size());
    EXPECT_EQ(15, network2->size());
}