#include "opencl/squared_error.h"

namespace mozart
{
    KERNEL(squared_error_kernel,
        __kernel void squared_error_kernel(              
                    __global TYPE * in,
                    __global TYPE * targets,
                    __local TYPE * local_buffer,
                    __global TYPE * out,
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

                TYPE diff = targets[internal_id] - in[internal_id];

                out[global_id] = diff*diff*0.5;
            }                                         
        }
    )

    KERNEL(squared_error_deriv_kernel,
                __kernel void squared_error_deriv_kernel(              
                    __global TYPE * in,
                    __global TYPE * targets,
                    __local TYPE * local_buffer,
                    __global TYPE * out,
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

                TYPE diff = in[internal_id] - targets[internal_id];

                out[global_id] = diff;
            }                                         
        }
    )

    namespace opencl
    {

        template<typename T>
        cost<T> squared_error(matrix<T>& in, matrix<T>& targets, bool derive)
        {
            cost<T> result(in, derive);

            kernel<T, squared_error_kernel>::instance()
                .with_global_size(in.total_size())
                .with_local_size(in.size2())
                .run(
                    in,
                    targets,
                    local<T>(in.size2()),
                    result.out,
                    in.size()
                );

            if(derive)
            {
                kernel<T, squared_error_deriv_kernel>::instance()
                    .with_global_size(in.total_size())
                    .with_local_size(in.size2())
                    .run(
                        in,
                        targets,
                        local<T>(in.size2()),
                        result.deriv,
                        in.size()
                    );
            }

            return result;
        }

        template cost<float> squared_error(matrix<float>& in, matrix<float>& targets, bool derive);
        template cost<double> squared_error(matrix<double>& in, matrix<double>& targets, bool derive);
    }
}
