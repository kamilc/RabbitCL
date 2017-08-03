#include "input_config.h"

using namespace std;

namespace heed {
    template<typename T, mode MODE>
    input_config<T, MODE>::input_config(size_t size) : _size(size) {}

    INSTANTIATE(input_config);
}