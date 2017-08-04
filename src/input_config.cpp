#include "input_config.h"

using namespace std;

namespace heed {
    template<typename T>
    input_config<T>::input_config(size_t size) : _size(size) {}

    INSTANTIATE(input_config);
}