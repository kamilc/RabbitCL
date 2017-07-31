#include <iostream>

#include "gtest/gtest.h"

#include <boost/numeric/ublas/matrix.hpp>

#include "input.h"
#include "dense.h"
#include "function/tanh.h"
#include "function/relu.h"
#include "function/softmax.h"
#include "matrix.h"
#include "gradient_descent.h"

using namespace heed;

namespace ublas = boost::numeric::ublas;

TEST(sample_test_case, sample_test)
{
    auto root = input<float, mode::cpu>::define(3);
    auto hidden = dense<float, mode::cpu>::define(3, root, function::tanh<float, mode::cpu>());
    auto output = dense<float, mode::cpu>::define(8, hidden, function::softmax<float, mode::cpu>());

    EXPECT_EQ(8, output->size());

    matrix<float> data = matrix<float>(mode::cpu, 8, 3, { 0, 0, 0, /* 0 */
                                                          0, 0, 1, /* 1 */
                                                          0, 1, 0, /* 2 */
                                                          0, 1, 1, /* 3 */
                                                          1, 0, 0, /* 4 */
                                                          1, 0, 1, /* 5 */
                                                          1, 1, 0, /* 6 */
                                                          1, 1, 1  /* 7 */ } );
    matrix<float> ys = matrix<float>(mode::cpu, 8, 8, { 1, 0, 0, 0, 0, 0, 0, 0, /* 0 */
                                                        0, 1, 0, 0, 0, 0, 0, 0, /* 1 */
                                                        0, 0, 1, 0, 0, 0, 0, 0, /* 2 */
                                                        0, 0, 0, 1, 0, 0, 0, 0, /* 3 */
                                                        0, 0, 0, 0, 1, 0, 0, 0, /* 4 */
                                                        0, 0, 0, 0, 0, 1, 0, 0, /* 5 */
                                                        0, 0, 0, 0, 0, 0, 1, 0, /* 6 */
                                                        0, 0, 0, 0, 0, 0, 0, 1 /* 7 */
                                                     });

    auto optimizer = gradient_descent<float, mode::cpu>()
                        .setEpochs(8*10)
                        .setBatches(1);
    
    optimizer.run(output);

    auto test = std::make_shared<matrix<float>>(matrix<float>(mode::cpu, 1, 3, { 0, 1, 0 }));
    auto expecting = matrix<float>(mode::cpu, 1, 8, { 0, 0, 1, 0, 0, 0, 0, 0 });

    auto predicted = output->forward(test);

    //EXPECT_EQ(predicted, expecting);
}