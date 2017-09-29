#include "function/categorical_cross_entropy.h"

#include <cmath>

namespace mozart
{
    KERNEL(categorical_cross_entropy_kernel,
        __kernel void categorical_cross_entropy_kernel(
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

                TYPE left = targets[internal_id];
                TYPE right = in[internal_id];

                TYPE diff = left - right;

                out[global_id] = left*log(right ? right : 0.00001);
            }
        }
    )

    KERNEL(categorical_cross_entropy_deriv_kernel,
                __kernel void categorical_cross_entropy_deriv_kernel(
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

    namespace function
    {

        template<typename T>
        cost<T> categorical_cross_entropy(matrix<T>& in, matrix<T>& targets, bool derive)
        {
            cost<T> result(in, derive);

            kernel<T, categorical_cross_entropy_kernel>::instance()
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
                kernel<T, categorical_cross_entropy_deriv_kernel>::instance()
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

           //for(auto row = 0; row < result.out.size1(); row++)
           //{
           //  for(auto col = 0; col < result.out.size2(); col++)
           //  {
           //    if(std::isnan(result.out(row, col)))
           //    {
           //      std::cout << "NaN detected!" << std::endl;
           //      std::cout << "Input: " << in << std::endl;
           //      std::cout << "Targets: " << targets << std::endl;
           //      std::cout << "Output: " << result.out << std::endl;
           //      assert(false == true);
           //    }
           //  }
           //}

            return result;
        }

        template cost<float> categorical_cross_entropy(matrix<float>& in, matrix<float>& targets, bool derive);
        template cost<double> categorical_cross_entropy(matrix<double>& in, matrix<double>& targets, bool derive);
    }
}
