#include "function/relu.h"

namespace mozart
{
    namespace function
    {
        // template<typename T>
        // inline const char * relu_kernel<T>::name()
        // {
        //     return "relu_kernel";
        // }

        // template<typename T>
        // inline const char * relu_kernel<T>::code()
        // {
        //     return  "__kernel void relu_kernel(\n"
        //             "          __global float * in,\n"
        //             "          __global float * out,\n"
        //             "              unsigned int size1,\n"
        //             "              unsigned int size2,\n"
        //             "              unsigned int isize1)\n"
        //             "{ \n"
        //             "  unsigned int gid = get_global_id(0);\n"
        //             "  unsigned int padded = isize1 - size1;\n"
        //             "  unsigned int row = gid / size2;\n"
        //             "  unsigned int idx = gid + row*padded;\n"
        //             "  out[idx] = fmax(0.0f, in[idx]);\n"
        //             "};\n";
        // }

        // template<typename T>
        // void relu_kernel<T>::compute_matrix(matrix<T>& in, matrix<T>& out)
        // {
        //     ocl::enqueue(this->_kernel(in,
        //                                out,
        //                                cl_uint(out.size1()),
        //                                cl_uint(out.size2()),
        //                                cl_uint(out.internal_size1())));
        //     finish();
        // }

        // template<typename T>
        // inline const char * relu_deriv_kernel<T>::name()
        // {
        //     return "relu_deriv_kernel";
        // }

        // template<typename T>
        // inline const char * relu_deriv_kernel<T>::code()
        // {
        //     return  "__kernel void relu_deriv_kernel(\n"
        //             "          __global float * in,\n"
        //             "          __global float * out,\n"
        //             "              unsigned int size1,\n"
        //             "              unsigned int size2,\n"
        //             "              unsigned int isize1)\n"
        //             "{ \n"
        //             "  unsigned int gid = get_global_id(0);\n"
        //             "  unsigned int padded = isize1 - size1;\n"
        //             "  unsigned int row = gid / size2;\n"
        //             "  unsigned int idx = gid + row*padded;\n"
        //             "  out[idx] = fmin(1.0f, floor(in[idx]));\n"
        //             "};\n";
        // }

        // template<typename T>
        // void relu_deriv_kernel<T>::compute_matrix(matrix<T>& in, matrix<T>& out)
        // {
        //     ocl::enqueue(this->_kernel(in,
        //                                out,
        //                                cl_uint(out.size1()),
        //                                cl_uint(out.size2()),
        //                                cl_uint(out.internal_size1())));
        //     finish();
        // }

        template<typename T>
        activation<T> relu(matrix<T>& in, bool derive)
        {
            activation<T> result(in, derive);

            //relu_kernel<T>::run_matrix(in, result.out, in.size1(), in.size2() * in.size1());

            if(derive)
            {
                //relu_deriv_kernel<T>::run_matrix(result.out, result.deriv, in.size1(), in.size2() * in.size1());
            }

            return result;
        }

        // INSTANTIATE(relu_kernel);
        // INSTANTIATE(relu_deriv_kernel);

        template activation<float> relu(matrix<float>& in, bool derive);
        template activation<double> relu(matrix<double>& in, bool derive);
    }
}