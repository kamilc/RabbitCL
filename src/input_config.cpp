#include "input_config.h"

using namespace std;

namespace mozart {
    template<typename T>
    input_config<T>::input_config(size_t size)
    {
        this->_size = size;
    }

    template<typename T>
    std::shared_ptr<layer<T>> input_config<T>::construct(std::size_t parent_size) const
    {
        auto _layer = make_shared<input<T>>();

        _layer->_size = this->_size;

        return _layer;
    }

    INSTANTIATE(input_config);
}