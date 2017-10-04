#ifndef Adagrad_h
#define Adagrad_h

#include <unordered_map>
#include <memory>
#include "gradient_descent.h"
#include "cost.h"
#include "matrix.h"
#include "utilities.h"
#include "sequence.h"
#include "function/adagrad_update.h"

using namespace std;
using namespace mozart::function;

namespace mozart
{
    namespace optimizers
    {
        template<typename T>
        class adagrad : public gradient_descent<T>
        {
        public:
            adagrad(typename cost<T>::function func);

            void update(size_t index, std::shared_ptr<layer<T>> layer, matrix<T>& deltas, matrix<T>& weight_deltas);

            adagrad& alpha(T alpha);
            adagrad& eps(T eps);
        private:
            matrix<T>& memo_for_index(size_t index, matrix<T>& deltas);

            unordered_map<size_t, std::shared_ptr<matrix<T>>> _memo;
            T _alpha = 0.01;
            T _eps = 1e-8;
        };
    }
}

#endif
