#include <iostream>
#include "sequence.h"
#include "input_config.h"
#include "dense_config.h"
#include "function/tanh.h"
#include "function/relu.h"
#include "function/softmax.h"
#include "function/squared_error.h"
#include "function/categorical_cross_entropy.h"
#include "optimizers/rmsprop.h"
#include "matrix_helpers.h"
#include "function/reduce_avg.h"
#include "function/dot.h"
#include "function/element_mul.h"
#include "function/squashmax.h"
#include "observer.h"
#include "observer/timed.h"
#include "stats/accuracy.h"

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION 1

using namespace mozart;
using namespace mozart::function;
using namespace mozart::stats;
using namespace mozart::observer;
using namespace mozart::optimizers;

template<typename T>
void expect_near(T value, T expected, T delta)
{
  assert(value - expected < delta);
}

int main()
{
    sequence<float> network;

    network.add(input_config<float>(3))
           .add(dense_config<float>(8, relu<float>))
           .add(dense_config<float>(8, relu<float>))
           .add(dense_config<float>(8, softmax<float>));

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

    rmsprop<float> optimizer(categorical_cross_entropy<float>);

    optimizer.epochs(1000)
             .batches(4)
             .push_observer(
                 timed<float>(std::chrono::seconds(1))
                     .stats(accuracy<float>)
                     .epoch_timing(true)
             );

    optimizer.run(network, data, ys);

    auto test = make_matrix<float>({{ 0, 1, 0 }});
    auto output = network.forward(test);
    auto predicted = squashmax<float>(output);

    std::cout << output << std::endl;
    std::cout << predicted << std::endl;

    expect_near<float>(predicted(0, 0), 0, 0.01);
    expect_near<float>(predicted(0, 1), 0, 0.01);
    expect_near<float>(predicted(0, 2), 1, 0.01);
    expect_near<float>(predicted(0, 3), 0, 0.01);
    expect_near<float>(predicted(0, 4), 0, 0.01);
    expect_near<float>(predicted(0, 5), 0, 0.01);
    expect_near<float>(predicted(0, 6), 0, 0.01);
    expect_near<float>(predicted(0, 7), 0, 0.01);

    return 0;
}
