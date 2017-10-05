#include "opencl/inplace_reduce_column_sum.h"

namespace mozart
{
    KERNEL(inplace_reduce_column_sum_kernel,
        __kernel void inplace_reduce_column_sum_kernel(
                    __global TYPE * inout,
                   const struct matrix_size size)
        {
            unsigned int column = get_global_id(0);

            // naive implementation
            // todo: implement it in a better way:

            local TYPE sum;
            sum = 0;
            barrier(CLK_LOCAL_MEM_FENCE);

            for(unsigned int row = 0; row < size.size1; row++)
            {
                sum += inout[size.internal_size2 * (size.start1 + row) + size.start2 + column];
            }

            inout[size.internal_size2 * size.start1 + size.start2 + column] = sum;
        };
    );

    namespace opencl
    {
        template<typename T>
        void inplace_reduce_column_sum(matrix<T>& inout)
        {
            kernel<T, inplace_reduce_column_sum_kernel>::instance()
                .with_global_size(inout.size2())
                // todo: infer as large of a localsize as possible
                .with_local_size(1)
                .run(
                    inout,
                    inout.size()
                );
        }

        template void inplace_reduce_column_sum(matrix<float>& inout);
        template void inplace_reduce_column_sum(matrix<double>& inout);
    }
}
