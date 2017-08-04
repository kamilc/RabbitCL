#ifndef LayerConfig_h
#define LayerConfig_h

#include <boost/optional.hpp>
#include <cstddef>

#include "utilities.h"

namespace mozart {

    template<typename T>
    class layer_config
    {
    protected:
        boost::optional<std::size_t> _input_size;
        std::size_t _size;
    public:
        void set_input_size(std::size_t size);
        std::size_t input_size();
        std::size_t size();
        bool has_inputs();
    };
}

#endif