#include "dense_config.h"
#include "function.h"

namespace heed {
    template<typename T>
    dense_config<T>::dense_config(std::size_t size, typename activation<T>::function fun)
    {
        this->_size = size;
        this->fun = fun;
    }

    INSTANTIATE(dense_config);
}