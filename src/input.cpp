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
    std::shared_ptr<matrix<T, MODE>> input<T, MODE>::forward(std::shared_ptr<matrix<T, MODE>> data)
    {
        return data;
    }

    template<typename T, mode MODE>
    std::shared_ptr<input<T, MODE>> input<T, MODE>::define(std::size_t size)
    {
        return std::make_shared<input<T, MODE>>(input<T, MODE>(size));
    }

    template class input<float, mode::cpu>;
    template class input<float, mode::gpu>;

    template class input<double, mode::cpu>;
    template class input<double, mode::gpu>;
}