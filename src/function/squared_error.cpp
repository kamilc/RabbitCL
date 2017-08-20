#include "function/squared_error.h"

namespace mozart
{
    namespace function
    {
        // template<typename T>
        // inline const char * kernel<T, "squared_error">::code()
        // {
        //     return  "__kernel void squared_error(                       \n"
        //             "          __global float * yhat,                   \n"
        //             "          __global float * targets,                \n"
        //             "          __global float * out,                    \n"
        //             "              unsigned int size1,                  \n"
        //             "              unsigned int size2,                  \n"
        //             "              unsigned int isize2)                 \n"
        //             "{                                                  \n"
        //             "  unsigned int global_id = get_global_id(0);       \n"
        //             "  unsigned int pad = isize2 - size2;               \n"
        //             "  unsigned int row = global_id / size2;            \n"
        //             "  unsigned int idx = gid + row * pad;              \n"
        //             "                                                   \n"
        //             "  float diff = targets[idx] - yhat[idx];           \n"
        //             "  out[idx] = 0.5*diff*diff;                        \n"
        //             "}";
        // }

        // template<typename T>
        // void squared_error_kernel<T>::compute_matrix(matrix<T>& in, matrix<T>& targets, scalar<T>& out)
        // {
        //     ocl::enqueue(this->_kernel(in,
        //                                targets,
        //                                out,
        //                                cl_uint(out.size1()),
        //                                cl_uint(out.size2()),
        //                                cl_uint(out.internal_size2())));
        //     finish();
        // }

        // template<typename T>
        // inline const char * squared_error_deriv_kernel<T>::name()
        // {
        //     return "squared_error_deriv_kernel";
        // }

        // template<typename T>
        // inline const char * squared_error_deriv_kernel<T>::code()
        // {
        //     // todo: add tests!
        //     return  "__kernel void squared_error_deriv_kernel(\n"
        //             "          __global float * yhat, \n"
        //             "          __global float * targets, \n"
        //             "          __global float * out,\n"
        //             "              unsigned int size1,\n"
        //             "              unsigned int size2,\n"
        //             "              unsigned int isize1)\n"
        //             "{ \n"
        //             "  unsigned int gid = get_global_id(0);\n"
        //             "  unsigned int padded = isize1 - size1;\n"
        //             "  unsigned int row = gid / size2;\n"
        //             "  unsigned int idx = gid + row*padded;\n"
        //             "  out[idx] = targets[idx] - yhat[idx];"
        //             "};\n";
        // }

        template<typename T>
        cost<T> squared_error(matrix<T>& in, matrix<T>& targets, bool derive)
        {
            cost<T> result(in, derive);

            // kernel<T, "squared_error">::instance()(in,
            //                                     targets,
            //                                     in.size1(),
            //                                     in.size2(),
            //                                     in.internal_size2());

            //squared_error_kernel<T>::run_matrix(in, targets, result.out, in.size1(), in.size2() * in.size1());

            // if(derive)
            // {
            //     squared_error_deriv_kernel<T>::run_matrix(in, targets, result.deriv, in.size1(), in.size2() * in.size1());
            // }

            return result;
        }

        template cost<float> squared_error(matrix<float>& in, matrix<float>& targets, bool derive);
        template cost<double> squared_error(matrix<double>& in, matrix<double>& targets, bool derive);
    }
}