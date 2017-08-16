#ifndef Sequence_h
#define Sequence_h

#include <list>
#include <memory>
#include <vector>
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "utilities.h"
#include "layer_config.h"
#include "layer.h"

using namespace std;
using namespace viennacl;

namespace mozart
{
    template<typename T>
    class sequence
    {
    private:
        std::list<std::shared_ptr<layer<T>>> _layers;
        size_t _last_layer_size = 0;
    public:
        sequence& add(const layer_config<T> &config);

        size_t size();

        virtual matrix<T> forward(matrix<T> &data);
        virtual std::vector<matrix<T>> train_forward(matrix_range<matrix<T>> &data);
    };
}

#endif