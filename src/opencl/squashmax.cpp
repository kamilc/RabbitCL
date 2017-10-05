#include "opencl/squashmax.h"

namespace mozart
{
    KERNEL(squashmax_kernel,
        __kernel void squashmax_kernel(
                   const __global TYPE * in,
                   __global TYPE * out,
                   const struct matrix_size in_size)
        {
            unsigned int row = get_global_id(0);
            unsigned int max_id = 0;
            TYPE last_max = 0;

            // todo: make this very naive implementation better:

            for(unsigned int iter = 0; iter < in_size.size2; iter++)
            {
              out[row * in_size.size2 + iter] = 0;

              TYPE current = in[row * in_size.size2 + iter];

              if(current > last_max)
              {
                last_max = current;
                max_id = iter;
              }
            }

            out[row * in_size.size2 + max_id] = 1;
        };
    );

    namespace opencl
    {
        template<typename T>
        matrix<T> squashmax(matrix<T>& in)
        {
            matrix<T> result(in.size1(), in.size2());

            kernel<T, squashmax_kernel>::instance()
                .with_global_size(in.total_size())
                .with_local_size(in.size1())
                .run(
                    in,
                    result,
                    in.size()
                );

            return result;
        }

        template matrix<float> squashmax(matrix<float>& in);
        template matrix<double> squashmax(matrix<double>& in);
    }
}

