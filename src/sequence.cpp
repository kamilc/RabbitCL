#include "sequence.h"

namespace mozart
{
    template<typename T>
    sequence<T>& sequence<T>::add(const layer_config<T> &config)
    {
        auto layer = config.construct(this->_last_layer_size);
        this->_layers.push_front(layer);

        this->_last_layer_size = config.size();

        return *this;
    }

    template<typename T>
    matrix<T> sequence<T>::forward(matrix<T> &data)
    {
        // todo: implement me
        return data;
    }

    template<typename T>
    std::vector<matrix<T>> sequence<T>::train_forward(matrix<T> &data)
    {
        std::vector<matrix<T>> out(this->size());

        out[0] = matrix<T>(data);

        for(auto index = 1; index < this->size(); index++)
        {
            out[index] = this->_layers[index]->forward(out[index - 1]);
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