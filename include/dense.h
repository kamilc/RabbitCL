#ifndef Dense_h
#define Dense_h

#include "utilities.h"
#include "layer.h"
#include "dense_config.h"
#include "function.h"

namespace mozart
{
    template<typename T>
    class dense : public layer<T>
    {
    public:
        dense(dense_config<T> &config);

        static dense_config<T> with(std::size_t size, typename activation<T>::function fun);

        matrix<T> forward(matrix<T> &data);
    private:
        typename activation<T>::function &_fun;
    };
}

#endif