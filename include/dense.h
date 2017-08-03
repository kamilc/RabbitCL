#ifndef Dense_h
#define Dense_h

#include "utilities.h"
#include "layer.h"
#include "dense_config.h"
#include "function.h"

namespace heed
{
    template<typename T, mode MODE>
    class dense : public layer<T, MODE>
    {
    public:
        dense(dense_config<T, MODE> &config);

        static dense_config<T, MODE> with(std::size_t size, typename activation<T, MODE>::function fun);

        matrix<T, MODE> forward(matrix<T, MODE> &data);
    private:
        typename activation<T, MODE>::function &_fun;
    };
}

#endif