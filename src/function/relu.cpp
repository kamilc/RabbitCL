#include "function/relu.h"

using namespace boost;

namespace mozart
{
    KERNEL(relu_kernel,
        __kernel void relu_kernel(
                    __global TYPE * in,
                    __global TYPE * out,
                   struct matrix_size in_size)
        {
            unsigned int global_id = get_global_id(0);
            unsigned int idx = id_to_internal_id(global_id, &in_size);

            out[idx] = fmax((TYPE)0.0, in[idx]);
        };
    );

    KERNEL(relu_deriv_kernel,
        __kernel void relu_deriv_kernel(
                  __global TYPE * in,
                  __global TYPE * out,
                 struct matrix_size in_size)
        {
            unsigned int global_id = get_global_id(0);
            unsigned int idx = id_to_internal_id(global_id, &in_size);

            out[idx] = fmin((TYPE)1.0, floor(in[idx]));
        };
    );

    namespace function
    {
        template<typename T>
        activation<T> relu(matrix<T>& in, bool derive)
        {
            activation<T> result(in, derive);

            kernel<T, relu_kernel>::instance()
                .with_global_size(in.total_size())
                // todo: infer as large of a localsize as possible
                .with_local_size(in.size2())
                .run(
                    in,
                    result.out,
                    in.size()
                );

            if(derive)
            {
                kernel<T, relu_deriv_kernel>::instance()
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

        template activation<float> relu(matrix<float>& in, bool derive);
        template activation<double> relu(matrix<double>& in, bool derive);
    }
}