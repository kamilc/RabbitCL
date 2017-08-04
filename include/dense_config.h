#ifndef DenseConfig_h
#define DenseConfig_h

#include <memory>
#include "utilities.h"
#include "layer_config.h"
#include "activation.h"
#include "dense.h"

using namespace std;
using namespace boost;

namespace mozart {
    template<typename T>
    class dense_config : public layer_config<T>
    {      
    public:
        typename activation<T>::function fun;

        dense_config(size_t size, typename activation<T>::function func);

        std::shared_ptr<layer<T>> construct() const;
    };
}

#endif