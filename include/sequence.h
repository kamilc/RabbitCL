#ifndef Sequence_h
#define Sequence_h

#include <deque>
#include <memory>
#include <vector>
#include "matrix.h"
#include "viennacl/matrix_proxy.hpp"
#include "utilities.h"
#include "layer_config.h"
#include "layer.h"
#include "activation.h"

using namespace std;
using namespace mozart;

namespace mozart
{
    template<typename T>
    class sequence
    {
    private:
        std::deque<std::shared_ptr<layer<T>>> _layers;
        size_t _last_layer_size = 0;
    public:
        sequence& add(const layer_config<T> &config);

        size_t size();

        std::shared_ptr<layer<T>> operator[](std::size_t index);

        virtual matrix<T> forward(matrix<T> &data);
        virtual std::vector<activation<T>> train_forward(matrix<T> &data);
    };
}

#endif