#include "gradient_descent.h"

using namespace std;

namespace mozart
{
    template<typename T>
    gradient_descent<T>::gradient_descent(typename cost<T>::function func)
    {
        this->_cost = func;
    }

    template<typename T>
    void gradient_descent<T>::run(sequence<T> &network, matrix<T> &data, matrix<T> &targets)
    {
        auto batches_len = data.size1() / this->_batches;
        auto columns_length = data.size2();

        for(auto epoch = 0; epoch < this->_epochs; epoch++)
        {
            T error = 0;

            for(auto batch = 0; batch < batches_len; batch++)
            {
                auto start = batch * this->_batches;
                auto end = start + this->_batches;
    
                auto batch_data = matrix<T>::view(data, start, end, 0, columns_length);
                auto batch_targets = matrix<T>::view(targets, start, end, 0, columns_length - 1);
    
                error = run_batch(network, batch_data, batch_targets);
            }

            std::cout << "Epoch \t[ " << epoch << " ] \t- " << error << std::endl;
        }
    }

    template<typename T>
    T gradient_descent<T>::run_batch(sequence<T> &network, matrix<T> &data, matrix<T> &targets)
    {
        // 1. get outputs of layers
        std::vector<matrix<T>> outputs = network.train_forward(data);
        matrix<T>& last_output = outputs[outputs.size() - 1];

        // 2. compute the network error
        cost<T> network_error = this->_cost(last_output, targets, true);

        // 3. compute little deltas
        std::vector<matrix<T>> deltas(outputs.size());

        deltas[outputs.size() - 1] = this->compute_last_deltas(last_output);
        for(auto layer_index = outputs.size() - 2; layer_index > 0; layer_index--)
        {
            deltas[layer_index] = this->compute_deltas(outputs[layer_index]);
        }

        // 4. compute Δw and update the weights on the fly
        //    this means multiplying the little delta by the input to the current layer
        //    which is the output of the previous one
        for(auto layer_index = outputs.size() - 1; layer_index > 0; layer_index--)
        {
            matrix<T>& delta = deltas[layer_index];
            matrix<T>& layer_input = outputs[layer_index - 1];

            auto weight_delta = matrix<T>(-1 * this->_eta * dot(delta, layer_input));
            network[layer_index]->update_weights(weight_delta);
        }

        // 5. return the error
        return network_error.avg();
    }

    template<typename T>
    inline matrix<T> gradient_descent<T>::compute_last_deltas(matrix<T>& outputs)
    {
        // todo: implement me
        return matrix<T>(1, 1);
    }

    template<typename T>
    inline matrix<T> gradient_descent<T>::compute_deltas(matrix<T>& outputs)
    {
        // todo: implement me
        return matrix<T>(1, 1);
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