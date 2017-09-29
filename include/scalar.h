#ifndef Scalar_h
#define Scalar_h

#include <boost/compute/container.hpp>
#include <memory>
#include "utilities.h"
#include "context_manager.h"

using namespace std;
using namespace boost;

namespace mozart
{
    template<typename T>
    class scalar
    {
    public:
        scalar();
        scalar(T initial);

        compute::array<T, 1>& data();

        void operator=(T value);
        operator T();
    private:
        std::shared_ptr<compute::array<T, 1>> _data;
    };
}

#endif
