#include "function/relu.h"

using namespace boost;

namespace mozart
{
    KERNEL(relu_kernel,
        template<typename T>
        __kernel void relu_kernel(
                    __global T * in,
                    __global T * out,
                   matrix_size * in_size,
                   matrix_size * out_size)
        {
            unsigned int gid = get_global_id(0);
            unsigned int padded = in_size.internal_size1 - in_size.size1;
            unsigned int row = gid / in_size.size2;
            unsigned int idx = gid + row * padded;

            out[idx] = fmax(0.0f, in[idx]);
        };

        template __attribute__((mangled_name(relu_kernel)))
        __kernel void relu_kernel(__global float * in,
                                  __global float * out,
                                     matrix_size * in_size,
                                     matrix_size * out_size);

        template __attribute__((mangled_name(relu_kernel)))
        __kernel void relu_kernel(__global double * in,
                                  __global double * out,
                                    matrix_size * in_size,
                                    matrix_size * out_size);
    );

    KERNEL(relu_deriv_kernel,
        template<typename T>
        __kernel void relu_deriv_kernel(
                  __global T * in,
                  __global T * out,
                 matrix_size * in_size,
                 matrix_size * out_size)
        {
            unsigned int gid = get_global_id(0);
            unsigned int padded = in_size.internal_size1 - in_size.size1;
            unsigned int row = gid / in_size.size2;
            unsigned int idx = gid + row * padded;

            out[idx] = fmin(1.0f, floor(in[idx]));
        };

        template __attribute__((mangled_name(relu_deriv_kernel)))
        __kernel void relu_deriv_kernel(__global float * in,
                                  __global float * out,
                                     matrix_size * in_size,
                                     matrix_size * out_size);

        template __attribute__((mangled_name(relu_deriv_kernel)))
        __kernel void relu_deriv_kernel(__global double * in,
                                  __global double * out,
                                    matrix_size * in_size,
                                    matrix_size * out_size);
    );

    namespace function
    {
        template<typename T>
        activation<T> relu(matrix<T>& in, bool derive)
        {
            activation<T> result(in, derive);

            kernel<T, relu_kernel>::instance().run(
                in.data(),
                result.out.data(),
                in.ocl_size(),
                result.out.ocl_size()
            );

            if(derive)
            {
                kernel<T, relu_deriv_kernel>::instance().run(
                    result.out.data(),
                    result.deriv.data(),
                    result.out.ocl_size(),
                    result.deriv.ocl_size()
                );
            }

            return result;
        }

        template activation<float> relu(matrix<float>& in, bool derive);
        template activation<double> relu(matrix<double>& in, bool derive);
    }
}