#include "gradient_descent.h"

using namespace std;

namespace heed
{
    template<typename T, mode MODE>
    gradient_descent<T, MODE>::gradient_descent()
    {
        // no-op
    }

    template<typename T, mode MODE>
    void gradient_descent<T, MODE>::run(layer<T, MODE> &network, matrix<T, MODE> &data, matrix<T, MODE> &targets)
    {
        for(auto epoch = 0; epoch < this->_epochs; epoch++)
        {
            cout << "Epoch " << epoch << endl;

            auto start = epoch * this->_batches;
            auto end = start + this->_batches - 1;

            auto batch_data = data.slice_rows(start, end);
            auto batch_targets = targets.slice_rows(start, end);

            run_batch(network, batch_data, batch_targets);
        }
    }

    template<typename T, mode MODE>
    void gradient_descent<T, MODE>::run_batch(layer<T, MODE> &network, matrix<T, MODE> &data, matrix<T, MODE> &targets)
    {
        // todo: implement me
    }

    template<typename T, mode MODE>
    gradient_descent<T, MODE>& gradient_descent<T, MODE>::setEpochs(unsigned long epochs)
    {
        this->_epochs = epochs;

        return *this;
    }

    template<typename T, mode MODE>
    gradient_descent<T, MODE>& gradient_descent<T, MODE>::setBatches(unsigned long batches)
    {
        this->_batches = batches;

        return *this;
    }

    template<typename T, mode MODE>
    gradient_descent<T, MODE>& gradient_descent<T, MODE>::setEta(T eta)
    {
        this->_eta = eta;

        return *this;
    }

    template class gradient_descent<float, mode::cpu>;
    template class gradient_descent<float, mode::gpu>;

    template class gradient_descent<double, mode::cpu>;
    template class gradient_descent<double, mode::gpu>;
}