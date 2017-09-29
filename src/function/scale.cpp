#include "function/scale.h"

namespace mozart
{
    KERNEL(scale_kernel,
        __kernel void scale_kernel(
                    __global TYPE * in,
                    const TYPE factor,
                    __global TYPE * out,
                   const struct matrix_size in_size)
        {
            unsigned int global_id = get_global_id(0);
            unsigned int idx = id_to_internal_id(global_id, &in_size);

            out[global_id] = in[idx] * factor;
        };
    );

    namespace function
    {
        template<typename T>
        matrix<T> scale(const matrix<T>& in, const T factor)
        {
            matrix<T> out(in.size1(), in.size2());

            kernel<T, scale_kernel>::instance()
                .with_global_size(in.total_size())
                // todo: infer as large of a localsize as possible
                .with_local_size(in.size2())
                .run(
                    in,
                    factor,
                    out,
                    in.size()
                );

            return out;
        }

        template matrix<float> scale(const matrix<float>& in, const float factor);
        template matrix<double> scale(const matrix<double>& in, const double factor);
    }
}
