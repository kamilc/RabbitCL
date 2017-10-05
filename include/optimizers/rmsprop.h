#ifndef Rmsprop_h
#define Rmsprop_h

#include <unordered_map>
#include <memory>
#include "gradient_descent.h"
#include "cost.h"
#include "matrix.h"
#include "utilities.h"
#include "sequence.h"
#include "opencl/rmsprop_update.h"

using namespace std;
using namespace mozart::opencl;

namespace mozart
{
    namespace optimizers
    {
        template<typename T>
        class rmsprop : public gradient_descent<T>
        {
        public:
            rmsprop(typename cost<T>::function func);

            void update(size_t index, std::shared_ptr<layer<T>> layer, matrix<T>& deltas, matrix<T>& weight_deltas);

            rmsprop& alpha(T alpha);
            rmsprop& eps(T eps);
            rmsprop& mu(T mu);
        private:
            matrix<T>& memo_for_index(size_t index, matrix<T>& deltas);

            unordered_map<size_t, std::shared_ptr<matrix<T>>> _memo;
            T _alpha = 0.01;
            T _mu = 0.99;
            T _eps = 1e-8;
        };
    }
}

#endif
