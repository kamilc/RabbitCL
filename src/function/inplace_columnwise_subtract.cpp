#include "function/inplace_columnwise_subtract.h"

namespace mozart
{
    KERNEL(inplace_columnwise_subtract_kernel,
        __kernel void inplace_columnwise_subtract_kernel(
                    __global TYPE * left,
                    const __global TYPE * right,
                   const struct matrix_size left_size,
                   const struct matrix_size right_size)
        {
            unsigned int global_id = get_global_id(0);
            unsigned int column = get_group_id(0);
            unsigned int row = get_local_id(0);

            // naive implementation
            // todo: implement me in a fast and proper way

            left[ left_size.internal_size2 * (left_size.start1 + row) + left_size.start2 + column ] -=
                right[ right_size.internal_size2 * right_size.start1 + right_size.start2 + column ];
        };
    );

    namespace function
    {
        template<typename T>
        void inplace_columnwise_subtract(matrix<T>& left, const matrix<T>& right)
        {
            // todo: implement me: add asserts

            kernel<T, inplace_columnwise_subtract_kernel>::instance()
                .with_global_size(left.total_size())
                // todo: infer as large of a localsize as possible
                .with_local_size(left.size1())
                .run(
                    left,
                    right,
                    left.size(),
                    right.size()
                );
        }

        template void inplace_columnwise_subtract(matrix<float>& left, const matrix<float>& right);
        template void inplace_columnwise_subtract(matrix<double>& left, const matrix<double>& right);
    }
}
