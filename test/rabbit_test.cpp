#include <iostream>

#include "gtest/gtest.h"

#include "input.h"

using namespace heed;

TEST(sample_test_case, sample_test)
{
    auto root = std::make_shared<input<float>>(input<float>(10));

    EXPECT_EQ(10, root->size());

    // boost::numeric::ublas::matrix<float> input = boost::numeric::ublas::matrix<float>(4, 100);
    // TanhActivation activation = TanhActivation();
    // TrainDef train = TrainDef(input);

    // layer root = layer(100, activation);
    // layer hidden = layer(root, 2, activation);
    // layer network = layer(hidden, 2, activation);

    // EXPECT_EQ(network.size(), 2);
    // EXPECT_EQ(network.totalSize(), 400);

    // network.train(train);

    //EXPECT_EQ(1, 2);
}