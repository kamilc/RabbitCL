#include <iostream>

#include "gtest/gtest.h"

#include "Layer.h"
#include "TanhActivation.h"
#include "TrainDef.h"

TEST(sample_test_case, sample_test)
{
    boost::numeric::ublas::matrix<float> input = boost::numeric::ublas::matrix<float>(100, 4);
    TanhActivation activation = TanhActivation();
    TrainDef train = TrainDef(input);

    Layer layer = Layer(100, activation);
    Layer hidden = Layer(layer, 2, activation);
    Layer network = Layer(hidden, 2, activation);

    EXPECT_EQ(network.size(), 2);
    EXPECT_EQ(network.totalSize(), 400);

    network.train(train);

    EXPECT_EQ(1, 2);
}