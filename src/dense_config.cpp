#include "dense_config.h"
#include "function.h"

namespace heed {
    template<typename T, mode MODE>
    dense_config<T, MODE>::dense_config(std::size_t size, typename activation<T, MODE>::function fun)
    {
        this->_size = size;
        this->fun = fun;
    }

    INSTANTIATE(dense_config);
}