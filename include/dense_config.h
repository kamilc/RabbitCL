#ifndef DenseConfig_h
#define DenseConfig_h

// #include <functional>
// #include <tuple>
// #include "boost/optional.hpp"

#include "utilities.h"
#include "layer_config.h"
#include "matrix.h"
#include "function.h"

using namespace std;
using namespace boost;

namespace heed {
    template<typename T, mode MODE>
    class dense_config : public layer_config<T, MODE>
    {      
    public:
        typename activation<T, MODE>::function fun;

        dense_config(size_t size, typename activation<T, MODE>::function func);
    };
}

#endif