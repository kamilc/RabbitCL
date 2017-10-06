#include <iostream>
#include "sequence.h"
#include "input_config.h"
#include "dense_config.h"
#include "opencl/tanh.h"
#include "opencl/relu.h"
#include "opencl/softmax.h"
#include "opencl/squared_error.h"
#include "opencl/categorical_cross_entropy.h"
#include "optimizers/adagrad.h"
#include "matrix_helpers.h"
#include "opencl/reduce_avg.h"
#include "opencl/dot.h"
#include "opencl/element_mul.h"
#include "opencl/squashmax.h"
#include "observer.h"
#include "observer/timed.h"
#include "stats/accuracy.h"

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION 1

using namespace mozart;
using namespace mozart::opencl;
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
           .add(dense_config<float>(8, tanh<float>))
           .add(dense_config<float>(8, tanh<float>))
           .add(dense_config<float>(8, tanh<float>))
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

    adagrad<float> optimizer(categorical_cross_entropy<float>);

    optimizer.epochs(5000)
             .batches(8)
             .push_observer(
                 timed<float>(std::chrono::seconds(1))
                     .stats(accuracy<float>)
                     .early_stop_when([](timed_observer<float>& observer){
                         if(observer._last_stat_value >= 1 && observer.last_error() < 0.04)
                         {
                            std::cout << "Early stopping with accuracy of " << observer._last_stat_value << std::endl;
                            return true;
                         }
                         return false;
                     })
                     .epoch_timing(true)
             );

    optimizer.run(network, data, ys);

    auto test = make_matrix<float>({
        {0, 0, 0},
        {0, 0, 1}, /* 1 */
        {0, 1, 0}, /* 2 */
        {0, 1, 1}, /* 3 */
        {1, 0, 0}, /* 4 */
        {1, 0, 1}, /* 5 */
        {1, 1, 0}, /* 6 */
        {1, 1, 1}  /* 7 */
    });
    auto output = network.forward(test);
    auto predicted = squashmax<float>(output);

    std::cout << output << std::endl;
    std::cout << predicted << std::endl;

    expect_near<float>(predicted(0, 0), 1, 0.01);
    expect_near<float>(predicted(0, 1), 0, 0.01);
    expect_near<float>(predicted(0, 2), 0, 0.01);
    expect_near<float>(predicted(0, 3), 0, 0.01);
    expect_near<float>(predicted(0, 4), 0, 0.01);
    expect_near<float>(predicted(0, 5), 0, 0.01);
    expect_near<float>(predicted(0, 6), 0, 0.01);
    expect_near<float>(predicted(0, 7), 0, 0.01);

    expect_near<float>(predicted(1, 0), 0, 0.01);
    expect_near<float>(predicted(1, 1), 1, 0.01);
    expect_near<float>(predicted(1, 2), 0, 0.01);
    expect_near<float>(predicted(1, 3), 0, 0.01);
    expect_near<float>(predicted(1, 4), 0, 0.01);
    expect_near<float>(predicted(1, 5), 0, 0.01);
    expect_near<float>(predicted(1, 6), 0, 0.01);
    expect_near<float>(predicted(1, 7), 0, 0.01);

    expect_near<float>(predicted(2, 0), 0, 0.01);
    expect_near<float>(predicted(2, 1), 0, 0.01);
    expect_near<float>(predicted(2, 2), 1, 0.01);
    expect_near<float>(predicted(2, 3), 0, 0.01);
    expect_near<float>(predicted(2, 4), 0, 0.01);
    expect_near<float>(predicted(2, 5), 0, 0.01);
    expect_near<float>(predicted(2, 6), 0, 0.01);
    expect_near<float>(predicted(2, 7), 0, 0.01);

    expect_near<float>(predicted(3, 0), 0, 0.01);
    expect_near<float>(predicted(3, 1), 0, 0.01);
    expect_near<float>(predicted(3, 2), 0, 0.01);
    expect_near<float>(predicted(3, 3), 1, 0.01);
    expect_near<float>(predicted(3, 4), 0, 0.01);
    expect_near<float>(predicted(3, 5), 0, 0.01);
    expect_near<float>(predicted(3, 6), 0, 0.01);
    expect_near<float>(predicted(3, 7), 0, 0.01);

    expect_near<float>(predicted(4, 0), 0, 0.01);
    expect_near<float>(predicted(4, 1), 0, 0.01);
    expect_near<float>(predicted(4, 2), 0, 0.01);
    expect_near<float>(predicted(4, 3), 0, 0.01);
    expect_near<float>(predicted(4, 4), 1, 0.01);
    expect_near<float>(predicted(4, 5), 0, 0.01);
    expect_near<float>(predicted(4, 6), 0, 0.01);
    expect_near<float>(predicted(4, 7), 0, 0.01);

    expect_near<float>(predicted(5, 0), 0, 0.01);
    expect_near<float>(predicted(5, 1), 0, 0.01);
    expect_near<float>(predicted(5, 2), 0, 0.01);
    expect_near<float>(predicted(5, 3), 0, 0.01);
    expect_near<float>(predicted(5, 4), 0, 0.01);
    expect_near<float>(predicted(5, 5), 1, 0.01);
    expect_near<float>(predicted(5, 6), 0, 0.01);
    expect_near<float>(predicted(5, 7), 0, 0.01);

    expect_near<float>(predicted(6, 0), 0, 0.01);
    expect_near<float>(predicted(6, 1), 0, 0.01);
    expect_near<float>(predicted(6, 2), 0, 0.01);
    expect_near<float>(predicted(6, 3), 0, 0.01);
    expect_near<float>(predicted(6, 4), 0, 0.01);
    expect_near<float>(predicted(6, 5), 0, 0.01);
    expect_near<float>(predicted(6, 6), 1, 0.01);
    expect_near<float>(predicted(6, 7), 0, 0.01);

    expect_near<float>(predicted(7, 0), 0, 0.01);
    expect_near<float>(predicted(7, 1), 0, 0.01);
    expect_near<float>(predicted(7, 2), 0, 0.01);
    expect_near<float>(predicted(7, 3), 0, 0.01);
    expect_near<float>(predicted(7, 4), 0, 0.01);
    expect_near<float>(predicted(7, 5), 0, 0.01);
    expect_near<float>(predicted(7, 6), 0, 0.01);
    expect_near<float>(predicted(7, 7), 1, 0.01);

    return 0;
}
