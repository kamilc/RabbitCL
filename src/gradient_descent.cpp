#include "gradient_descent.h"

using namespace std;

namespace mozart
{
    template<typename T>
    gradient_descent<T>::gradient_descent()
    {
        // no-op
    }

    template<typename T>
    void gradient_descent<T>::run(sequence<T> &network, matrix<T> &data, matrix<T> &targets)
    {
        for(auto epoch = 0; epoch < this->_epochs; epoch++)
        {
            // auto start = epoch * this->_batches;
            // auto end = start + this->_batches - 1;

            // auto batch_data = data.slice_rows(start, end);
            // auto batch_targets = targets.slice_rows(start, end);

            // run_batch(network, batch_data, batch_targets);

            // todo: implement me
        }
    }

    template<typename T>
    void gradient_descent<T>::run_batch(sequence<T> &network, matrix<T> &data, matrix<T> &targets)
    {
        // todo: implement me
    }

    template<typename T>
    gradient_descent<T>& gradient_descent<T>::setEpochs(unsigned long epochs)
    {
        this->_epochs = epochs;

        return *this;
    }

    template<typename T>
    gradient_descent<T>& gradient_descent<T>::setBatches(unsigned long batches)
    {
        this->_batches = batches;

        return *this;
    }

    template<typename T>
    gradient_descent<T>& gradient_descent<T>::setEta(T eta)
    {
        this->_eta = eta;

        return *this;
    }

    INSTANTIATE(gradient_descent);
}