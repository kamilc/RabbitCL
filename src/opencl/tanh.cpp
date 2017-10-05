#include "opencl/tanh.h"

namespace mozart
{
    KERNEL(tanh_kernel,
        __kernel void tanh_kernel(
                    __global TYPE * in,
                    __global TYPE * out,
                   const struct matrix_size in_size)
        {
            unsigned int global_id = get_global_id(0);
            unsigned int idx = id_to_internal_id(global_id, &in_size);

            out[idx] = tanh(in[idx]);
        };
    );

    KERNEL(tanh_deriv_kernel,
        __kernel void tanh_deriv_kernel(
                  __global TYPE * in,
                  __global TYPE * out,
                 const struct matrix_size in_size)
        {
            unsigned int global_id = get_global_id(0);
            unsigned int idx = id_to_internal_id(global_id, &in_size);

            out[global_id] = 1 - in[idx] * in[idx];
        };
    );

    namespace opencl
    {
        template<typename T>
        activation<T> tanh(matrix<T>& in, bool derive)
        {
            activation<T> result(in, derive);

            kernel<T, tanh_kernel>::instance()
                .with_global_size(in.total_size())
                // todo: infer as large of a localsize as possible
                .with_local_size(in.size2())
                .run(
                    in,
                    result.out,
                    in.size()
                );

            if(derive) {
                kernel<T, tanh_deriv_kernel>::instance()
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

        template activation<float> tanh(matrix<float>& in, bool derive);
        template activation<double> tanh(matrix<double>& in, bool derive);
    }
}
