#include "opencl/accuracy_rate.h"

namespace mozart
{
    KERNEL(accuracy_rate_kernel,
        __kernel void accuracy_rate_kernel(
            __global TYPE * predicted,
            __global TYPE * targets,
            const struct matrix_size size,
            __local TYPE * results,
            __global TYPE * out
        )
        {
            unsigned int global_id = get_global_id(0);
            unsigned int local_id = get_local_id(0);
            unsigned int group_id = get_group_id(0);
            unsigned int idx = id_to_internal_id(global_id, &size);

            __local bool correct;
            correct = true;
            barrier(CLK_LOCAL_MEM_FENCE);

            if(predicted[idx] != targets[idx])
            {
                correct = false;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if(local_id == 0)
            {
                results[group_id] = correct ? 1.0 : 0.0;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            if(global_id == 0)
            {
                TYPE sum = 0;
                for(unsigned int i = 0; i < size.size1; i++)
                {
                    sum += results[i];
                }
                out[0] = sum / (TYPE)size.size1;
            }
        }
    );
    namespace opencl
    {
        template<typename T>
        scalar<T> accuracy_rate(const matrix<T>& predicted, const matrix<T>& targets)
        {
            scalar<T> out;

            kernel<T, accuracy_rate_kernel>::instance()
                .with_global_size(predicted.total_size())
                // todo: infer as large of a localsize as possible
                .with_local_size(predicted.size2())
                .run(
                    predicted,
                    targets,
                    predicted.size(),
                    local<T>(predicted.size1()),
                    out
                );

            return out;
        }

        template scalar<float> accuracy_rate(const matrix<float>& predicted, const matrix<float>& targets);
        template scalar<double> accuracy_rate(const matrix<double>& predicted, const matrix<double>& targets);
    }
}
