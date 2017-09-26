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
        try
        {
            for(auto i = 0; i < this->_reporters.size(); i++)
            {
                this->_reporters[i]->start();
            }

            auto batches_len = data.size1() / this->_batches;
            auto columns_length = data.size2();
            auto targets_columns_length = targets.size2();

            for(auto epoch = 0; epoch < this->_epochs; epoch++)
            {
                for(auto i = 0; i < this->_reporters.size(); i++)
                {
                    this->_reporters[i]->start_epoch(epoch, this->_epochs);
                }

                for(auto batch = 0; batch < batches_len; batch++)
                {
                    for(auto i = 0; i < this->_reporters.size(); i++)
                    {
                      this->_reporters[i]->start_batch(batch, batches_len);
                    }

                    auto start = batch * this->_batches;
                    auto end = start + this->_batches - 1;

                    auto batch_data = matrix<T>::view(data, start, end, 0, columns_length - 1);
                    auto batch_targets = matrix<T>::view(targets, start, end, 0, targets_columns_length - 1);

                    run_batch(network, batch_data, batch_targets);

                    for(auto i = 0; i < this->_reporters.size(); i++)
                    {
                      this->_reporters[i]->end_batch();
                    }
                }

                for(auto i = 0; i < this->_reporters.size(); i++)
                {
                  this->_reporters[i]->end_epoch(network);
                }
            }

            for(auto i = 0; i < this->_reporters.size(); i++)
            {
                this->_reporters[i]->end();
            }

        }
        catch(std::exception& e)
        {
            for(auto i = 0; i < this->_reporters.size(); i++)
            {
                this->_reporters[i]->end();
            }
        }
    }

    template<typename T>
    void gradient_descent<T>::run_batch(sequence<T> &network, matrix<T> &data, matrix<T> &targets)
    {
        // 1. get outputs of layers
        std::vector<activation<T>> outputs = network.train_forward(data);
        activation<T>& last_output = outputs[outputs.size() - 1];

        // 2. compute the network error
        cost<T> network_error = this->_cost(last_output.out, targets, true);

        if(network_error.avg() * 0.0 != 0.0)
        {
            std::cout << "Got nan in network error" << std::endl;
            std::cout << "Error matrix: " << network_error.out << std::endl;
            std::cout << "Data matrix: " << data << std::endl;
            std::cout << "Targets: " << targets << std::endl;

            for(unsigned int i = 0; i < outputs.size(); i++)
            {
                std::cout << "Output #" << i << std::endl;
                std::cout << outputs[i].out;
                std::cout << "Deriv #" << i << std::endl;
                std::cout << outputs[i].deriv;
                std::cout << "Weights #" << i << std::endl;
                std::cout << network[i]->weights() << std::endl;
            }
            assert(network_error.avg() * 0.0 == 0.0);
        }

        // 3. compute little deltas
        std::vector<matrix<T>> deltas(outputs.size());

        deltas[outputs.size() - 1] = network_error.deriv * last_output.deriv;

        for(auto layer_index = outputs.size() - 2; layer_index > 0; layer_index--)
        {
            auto pullback = dot(deltas[layer_index + 1], network[layer_index + 1]->weights(), false, true);
            deltas[layer_index] = pullback * outputs[layer_index].deriv;
        }

        // 4. compute Δw and update the weights on the fly
        //    this means multiplying the little delta by the input to the current layer
        //    which is the output of the previous one
        for(auto layer_index = outputs.size() - 1; layer_index > 0; layer_index--)
        {
            matrix<T>& delta = deltas[layer_index];
            matrix<T>& layer_input = outputs[layer_index - 1].out;

            auto weight_delta = matrix<T>(-1 * this->_eta * dot(layer_input, delta, true, false));
            network[layer_index]->update_bias(delta);
            network[layer_index]->update_weights(weight_delta);
        }

        for(auto i = 0; i < this->_reporters.size(); i++)
        {
            this->_reporters[i]->push_outputs(last_output, targets);
            this->_reporters[i]->push_error(network_error);
        }
    }

    template<typename T>
    gradient_descent<T>& gradient_descent<T>::epochs(unsigned long epochs)
    {
        this->_epochs = epochs;

        return *this;
    }

    template<typename T>
    gradient_descent<T>& gradient_descent<T>::batches(unsigned long batches)
    {
        this->_batches = batches;

        return *this;
    }

    template<typename T>
    gradient_descent<T>& gradient_descent<T>::push_reporter(mozart::reporter::config<T>& config)
    {
        this->_reporters.push_back(std::move(config.construct()));

        return *this;
    }

    template<typename T>
    gradient_descent<T>& gradient_descent<T>::eta(T eta)
    {
        this->_eta = eta;

        return *this;
    }

    INSTANTIATE(gradient_descent);
}
