#include <iostream>

#include "gtest/gtest.h"

#include "Layer.h"
#include "TanhActivation.h"

TEST(sample_test_case, sample_test)
{
    TanhActivation activation = TanhActivation();

    Layer layer = Layer(100, activation);
    Layer network = Layer(layer, 2, activation);

    EXPECT_EQ(network.size(), 2);
    EXPECT_EQ(network.totalSize(), 200);
}