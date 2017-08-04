#include "input_config.h"

using namespace std;

namespace mozart {
    template<typename T>
    input_config<T>::input_config(size_t size) : _size(size) {}

    template<typename T>
    std::shared_ptr<layer<T>> input_config<T>::construct() const
    {
        auto ptr = make_shared<input<T>>();

        // todo:

        return ptr;
    }

    INSTANTIATE(input_config);
}