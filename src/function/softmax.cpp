#include "function/softmax.h"

namespace mozart
{
    KERNEL(softmax_kernel,
        __kernel void softmax_kernel(
                    __global TYPE * in,
                    __global TYPE * out,
                    __local TYPE * local_buffer,
                   const struct matrix_size in_size)
        {
            unsigned int global_id  = get_global_id(0);
            unsigned int total_size = in_size.size1 * in_size.size2;

            if(global_id < total_size)                
            {                                         
                unsigned int local_id = get_local_id(0);
                unsigned int group_id = get_group_id(0);
                unsigned int group_size = get_local_size(0);
                unsigned int internal_id = id_to_internal_id(global_id, &in_size);
                
                local_buffer[local_id] = exp(in[internal_id]);

                barrier(CLK_LOCAL_MEM_FENCE);

                for(int i = ( group_size + 1 ) / 2; i > 0; i >>= 1)
                {
                    if(local_id < i)
                    {
                        local_buffer[local_id] += local_buffer[local_id + i];
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }

                out[internal_id] = exp(in[internal_id]) / local_buffer[0];
            }          
        };
    );

    KERNEL(softmax_deriv_kernel,
        __kernel void softmax_deriv_kernel(
                  __global TYPE * in,
                  __global TYPE * out,
                 const struct matrix_size in_size)
        {
            unsigned int global_id = get_global_id(0);
            unsigned int idx = id_to_internal_id(global_id, &in_size);

            out[idx] = in[idx] * (1 - in[idx]);
        };
    );

    namespace function
    {
        template<typename T>
        activation<T> softmax(matrix<T>& in, bool derive)
        {
            activation<T> result(in, derive);

            kernel<T, softmax_kernel>::instance()
                .with_global_size(in.total_size())
                .with_local_size(in.size2())
                .run(
                    in,
                    result.out,
                    local<T>(in.size2()),
                    in.size()
                );

            if(derive) {
                kernel<T, softmax_deriv_kernel>::instance()
                    .with_global_size(in.total_size())
                    // todo: infer as large of a localsize as possible
                    .with_local_size(in.size2())
                    .run(
                        result.out,
                        result.deriv,
                        result.out.size()
                    );
            }

            return result;
        }

        template activation<float> softmax(matrix<float>& in, bool derive);
        template activation<double> softmax(matrix<double>& in, bool derive);
    }
}