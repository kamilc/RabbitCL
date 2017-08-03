#ifndef InputConfig_h
#define InputConfig_h

#include "utilities.h"
#include "layer_config.h"
#include "matrix.h"

namespace heed {
    template<typename T, mode MODE>
    class input_config : public layer_config<T, MODE>
    {
    private:
        std::size_t _size;
    public:
        input_config(std::size_t size);
    };
}

#endif