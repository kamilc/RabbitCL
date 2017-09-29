#include "sequence.h"

namespace mozart
{
    template<typename T>
    sequence<T>& sequence<T>::add(const layer_config<T> &config)
    {
        auto layer = config.construct(this->_last_layer_size);
        this->_layers.push_back(layer);

        this->_last_layer_size = config.size();

        return *this;
    }

    template<typename T>
    matrix<T> sequence<T>::forward(matrix<T> &data)
    {
        // todo: provide proper non train version
        std::vector<activation<T>> out(this->size());
        
        out[0] = this->_layers[0]->train_forward(data);

        for(auto index = 1; index < this->size(); index++)
        {
            out[index] = this->_layers[index]->train_forward(out[index - 1].out);
        }

        return out[this->size() - 1].out;
    }

    template<typename T>
    std::vector<activation<T>> sequence<T>::train_forward(matrix<T> &data)
    {
        std::vector<activation<T>> out(this->size());

        out[0] = this->_layers[0]->train_forward(data);

        for(auto index = 1; index < this->size(); index++)
        {
            out[index] = this->_layers[index]->train_forward(out[index - 1].out);
        }

        return out;
    }

    template<typename T>
    size_t sequence<T>::size()
    {
        return this->_layers.size();
    }

    template<typename T>
    std::shared_ptr<layer<T>> sequence<T>::operator[](std::size_t index)
    {
        return this->_layers[index];
    }

    INSTANTIATE(sequence);
}
