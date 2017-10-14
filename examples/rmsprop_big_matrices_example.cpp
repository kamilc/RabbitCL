#include <iostream>
#include "sequence.h"
#include "input_config.h"
#include "dense_config.h"
#include "opencl/tanh.h"
#include "opencl/relu.h"
#include "opencl/softmax.h"
#include "opencl/squared_error.h"
#include "opencl/categorical_cross_entropy.h"
#include "optimizers/rmsprop.h"
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

    network.add(input_config<float>(128))
           .add(dense_config<float>(128, tanh<float>))
           .add(dense_config<float>(128, tanh<float>))
           .add(dense_config<float>(128, tanh<float>))
           .add(dense_config<float>(10, softmax<float>));

    auto data = matrix<float>(12800, 128);
    data.fill_randn(0, 1);

    auto ys = matrix<float>(12800, 10);
    ys.fill_randn(0, 1);

    rmsprop<float> optimizer(categorical_cross_entropy<float>);

    optimizer.epochs(1000)
             .batches(128)
             .push_observer(
                 timed<float>(std::chrono::seconds(1))
                     .stats(accuracy<float>)
                     .epoch_timing(true)
             );

    optimizer.run(network, data, ys);

    // ending here as this example is solely to see how fast
    // the optimizer goes through data

    return 0;
}
