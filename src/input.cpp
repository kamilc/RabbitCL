#include "input.h"
#include "matrix.h"

namespace heed
{
    template<typename T, mode MODE>
    input<T, MODE>::input(std::size_t size) : layer<T, MODE>(size)
    {
        // no-op: implemented
    }

    template<typename T, mode MODE>
    void input<T, MODE>::forward(matrix<T, MODE> &data, matrix<T, MODE> &out)
    {
        // todo: implement me
    }

    template class input<float, mode::cpu>;
    template class input<float, mode::gpu>;

    template class input<double, mode::cpu>;
    template class input<double, mode::gpu>;
}