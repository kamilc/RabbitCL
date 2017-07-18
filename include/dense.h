#ifndef Dense_h
#define Dense_h

#include "layer.h"

namespace heed
{
    template<typename T>
    class dense : public layer<T>
    {
    public:
        dense(std::size_t size, std::shared_ptr<layer<T>> input);
        dense(std::size_t size, std::vector<std::shared_ptr<layer<T>>> inputs);

        std::shared_ptr<matrix<T>> forward(std::shared_ptr<matrix<T>> data);
    };
}

#endif