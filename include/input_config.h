#ifndef InputConfig_h
#define InputConfig_h

#include <memory>
#include "utilities.h"
#include "layer_config.h"
#include "layer.h"
#include "input.h"

using namespace std;

namespace mozart {
    template<typename T>
    class input_config : public layer_config<T>
    {
    public:
        input_config(std::size_t size);

        std::shared_ptr<layer<T>> construct(std::size_t parent_size) const;
    };
}

#endif