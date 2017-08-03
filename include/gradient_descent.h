#ifndef GradientDescent_h
#define GradientDescent_h

#include "utilities.h"
#include "optimizer.h"
#include "matrix.h"
#include "sequence.h"

namespace heed
{
    template<typename T, mode MODE>
    class gradient_descent : optimizer<T, MODE>
    {
    public:
        gradient_descent();

        gradient_descent& setEpochs(unsigned long epochs);
        gradient_descent& setBatches(unsigned long batches);
        gradient_descent& setEta(T eta);

        void run(sequence<T, MODE> &network, matrix<T, MODE> &data, matrix<T, MODE> &targets);
        void run_batch(sequence<T, MODE> &network, matrix<T, MODE> &data, matrix<T, MODE> &targets);
    private:
        T _eta;
        unsigned long  _epochs;
        unsigned long  _batches;
    };
}

#endif