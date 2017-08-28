#ifndef GradientDescent_h
#define GradientDescent_h

#include "matrix.h"
#include "utilities.h"
#include "optimizer.h"
#include "sequence.h"
#include "cost.h"
#include "function/dot.h"

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
        T run_batch(sequence<T> &network, matrix<T> &data, matrix<T> &targets);
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