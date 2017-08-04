#ifndef GradientDescent_h
#define GradientDescent_h

#include "viennacl/matrix.hpp"
#include "utilities.h"
#include "optimizer.h"
#include "sequence.h"

using namespace viennacl;

namespace mozart
{
    template<typename T>
    class gradient_descent : optimizer<T>
    {
    public:
        gradient_descent();

        gradient_descent& setEpochs(unsigned long epochs);
        gradient_descent& setBatches(unsigned long batches);
        gradient_descent& setEta(T eta);

        void run(sequence<T> &network, matrix<T> &data, matrix<T> &targets);
        void run_batch(sequence<T> &network, matrix<T> &data, matrix<T> &targets);
    private:
        T _eta;
        unsigned long  _epochs;
        unsigned long  _batches;
    };
}

#endif