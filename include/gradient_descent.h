#ifndef GradientDescent_h
#define GradientDescent_h

#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "utilities.h"
#include "optimizer.h"
#include "sequence.h"
#include "cost.h"

using namespace viennacl;
using namespace std;

namespace mozart
{
    template<typename T>
    class gradient_descent : optimizer<T>
    {
    public:
        gradient_descent(typename cost<T>::function func);

        gradient_descent& setEpochs(unsigned long epochs);
        gradient_descent& setBatches(unsigned long batches);
        gradient_descent& setEta(T eta);

        void run(sequence<T> &network, matrix<T> &data, matrix<T> &targets);
        T run_batch(sequence<T> &network, matrix_range<matrix<T>> &data, matrix_range<matrix<T>> &targets);
    private:
        T _eta;
        unsigned long  _epochs;
        unsigned long  _batches;
        typename cost<T>::function _cost;

        inline matrix<T> compute_last_deltas(matrix<T>& outputs);
        inline matrix<T> compute_deltas(matrix<T>& outputs);
    };
}

#endif