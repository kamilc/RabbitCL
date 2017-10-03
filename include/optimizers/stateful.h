#ifndef OptimizerStateful_h
#define OptimizerStateful_h

#include <vector>
#include <memory>
#include "matrix.h"
#include "utilities.h"
#include "sequence.h"
#include "optimizer.h"

using namespace std;

namespace mozart
{
    namespace optimizers
    {
        template<typename T, int N>
        class stateful : optimizer<T>
        {
            virtual void run(sequence<T> &network, matrix<T> &data, matrix<T> &targets) = 0;

        private:
            std::vector<std::shared_ptr<matrix<T>>> _memory[N];
        };
    }
}

#endif
