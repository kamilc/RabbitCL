#ifndef LayerConfig_h
#define LayerConfig_h

#include <boost/optional.hpp>
#include <cstddef>
#include <memory>

#include "utilities.h"
#include "layer.h"

using namespace std;

namespace mozart {
    template<typename T>
    class layer_config
    {
    protected:
        std::size_t _size;
    public:
        std::size_t size() const;

        virtual std::shared_ptr<layer<T>> construct(std::size_t parent_size) const = 0;
    };
}

#endif
