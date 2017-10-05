#ifndef GradientDescent_h
#define GradientDescent_h

#include <memory>
#include <vector>
#include <exception>
#include "matrix.h"
#include "utilities.h"
#include "optimizer.h"
#include "sequence.h"
#include "layer.h"
#include "cost.h"
#include "opencl/dot.h"
#include "observer.h"

using namespace std;

namespace mozart
{
    namespace optimizers
    {
        template<typename T>
        class gradient_descent : public optimizer<T>
        {
        public:
            gradient_descent(typename cost<T>::function func);

            gradient_descent& eta(T eta);
            void run(sequence<T> &network, matrix<T> &data, matrix<T> &targets);
            void run_batch(sequence<T> &network, matrix<T> &data, matrix<T> &targets);
            virtual void update(size_t index, std::shared_ptr<layer<T>> layer, matrix<T>& deltas, matrix<T>& weight_deltas);
        private:
            T _eta;
        };
    }
}

#endif
