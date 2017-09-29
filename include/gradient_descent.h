#ifndef GradientDescent_h
#define GradientDescent_h

#include <memory>
#include <vector>
#include <exception>
#include "matrix.h"
#include "utilities.h"
#include "optimizer.h"
#include "sequence.h"
#include "cost.h"
#include "function/dot.h"
#include "observer.h"

using namespace std;

namespace mozart
{
    template<typename T>
    class gradient_descent : optimizer<T>
    {
    public:
        gradient_descent(typename cost<T>::function func);

        gradient_descent& epochs(unsigned long epochs);
        gradient_descent& batches(unsigned long batches);
        gradient_descent& eta(T eta);
        gradient_descent& push_observer(mozart::observer::config<T>& config);

        void run(sequence<T> &network, matrix<T> &data, matrix<T> &targets);
        void run_batch(sequence<T> &network, matrix<T> &data, matrix<T> &targets);
    private:
        T _eta;
        unsigned long  _epochs;
        unsigned long  _batches;
        typename cost<T>::function _cost;
        vector<unique_ptr<mozart::observer::base<T>>> _observers;
    };
}

#endif
