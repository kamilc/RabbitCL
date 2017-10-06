#include "optimizers/gradient_descent.h"

using namespace std;

namespace mozart
{
    namespace optimizers
    {
        template<typename T>
        gradient_descent<T>::gradient_descent(typename cost<T>::function func)
        {
            this->_cost = func;
        }

        template<typename T>
        void gradient_descent<T>::run(sequence<T> &network, matrix<T> &data, matrix<T> &targets)
        {
            bool should_continue = true;

            try
            {
                for(auto i = 0; i < this->_observers.size(); i++)
                {
                    this->_observers[i]->start();
                }

                auto batches_len = data.size1() / this->_batches;
                auto columns_length = data.size2();
                auto targets_columns_length = targets.size2();

                for(auto epoch = 0; should_continue && (epoch < this->_epochs); epoch++)
                {
                    for(auto i = 0; i < this->_observers.size(); i++)
                    {
                        this->_observers[i]->start_epoch(epoch, this->_epochs);
                    }

                    for(auto batch = 0; batch < batches_len; batch++)
                    {
                        for(auto i = 0; i < this->_observers.size(); i++)
                        {
                          this->_observers[i]->start_batch(batch, batches_len);
                        }

                        auto start = batch * this->_batches;
                        auto end = start + this->_batches - 1;

                        auto batch_data = matrix<T>::view(data, start, end, 0, columns_length - 1);
                        auto batch_targets = matrix<T>::view(targets, start, end, 0, targets_columns_length - 1);

                        run_batch(network, batch_data, batch_targets);

                        for(auto i = 0; i < this->_observers.size(); i++)
                        {
                          this->_observers[i]->end_batch();
                        }
                    }

                    for(auto i = 0; i < this->_observers.size(); i++)
                    {
                        should_continue = this->_observers[i]->end_epoch(network);
                    }
                }

                for(auto i = 0; i < this->_observers.size(); i++)
                {
                    this->_observers[i]->end();
                }

            }
            catch(std::exception& e)
            {
                std::cout << "Error ocurred inside an optimizer! - " << e.what() << std::endl;

                for(auto i = 0; i < this->_observers.size(); i++)
                {
                    this->_observers[i]->end();
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

                auto weight_delta = dot(layer_input, delta, true, false);
                this->update(layer_index, network[layer_index], delta, weight_delta);
            }

            for(auto i = 0; i < this->_observers.size(); i++)
            {
                this->_observers[i]->push_outputs(last_output, targets);
                this->_observers[i]->push_error(network_error);
            }
        }

        template<typename T>
        void gradient_descent<T>::update(size_t index, std::shared_ptr<layer<T>> layer, matrix<T>& delta, matrix<T>& weight_delta)
        {
            auto update_delta = matrix<T>(-1 * this->_eta * weight_delta);

            layer->update_bias(delta);
            layer->update_weights(update_delta);
        }

        template<typename T>
        gradient_descent<T>& gradient_descent<T>::eta(T eta)
        {
            this->_eta = eta;

            return *this;
        }

        INSTANTIATE(gradient_descent);
    }
}
