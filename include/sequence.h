#ifndef Sequence_h
#define Sequence_h

#include <list>
#include <memory>
#include "viennacl/matrix.hpp"
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
    public:
        sequence& add(const layer_config<T> &config);

        size_t size();

        virtual matrix<T> forward(matrix<T> &data);
    };
}

#endif