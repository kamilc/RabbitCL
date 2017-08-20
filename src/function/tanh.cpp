#include "function/tanh.h"

namespace mozart
{
    namespace function
    {
        // template<typename T>
        // inline const char * tanh_kernel<T>::name()
        // {
        //     return "tanh_kernel";
        // }

        // template<typename T>
        // inline const char * tanh_kernel<T>::code()
        // {
        //     return  "__kernel void tanh_kernel(\n"
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
        //             "  out[idx] = tanh(in[idx]);\n"
        //             "};\n";
        // }

        // template<typename T>
        // void tanh_kernel<T>::compute_matrix(matrix<T>& in, matrix<T>& out)
        // {
        //     ocl::enqueue(this->_kernel( in,
        //                                 out,
        //                                 cl_uint(out.size1()),
        //                                 cl_uint(out.size2()),
        //                                 cl_uint(out.internal_size1())));
        //     finish();
        // }

        // template<typename T>
        // inline const char * tanh_deriv_kernel<T>::name()
        // {
        //     return "tanh_deriv_kernel";
        // }

        // template<typename T>
        // inline const char * tanh_deriv_kernel<T>::code()
        // {
        //     return  "__kernel void tanh_deriv_kernel(\n"
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
        //             "  out[idx] = 1.0 - in[idx] * in[idx];\n"
        //             "};\n";
        // }

        // template<typename T>
        // void tanh_deriv_kernel<T>::compute_matrix(matrix<T>& in, matrix<T>& out)
        // {
        //     ocl::enqueue(this->_kernel( in,
        //                                 out,
        //                                 cl_uint(out.size1()),
        //                                 cl_uint(out.size2()),
        //                                 cl_uint(out.internal_size1())));
        //     finish();
        // }

        // todo: provide the activation function also for the
        // speedy native_ variants

        template<typename T>
        activation<T> tanh(matrix<T>& in, bool derive)
        {
            activation<T> result(in, derive);

            //tanh_kernel<T>::run_matrix(in, result.out, in.size1(), in.size2() * in.size1());

            if(derive) {
                //tanh_deriv_kernel<T>::run_matrix(result.out, result.deriv, in.size1(), in.size2() * in.size1());
            }

            return result;
        }

        // INSTANTIATE(tanh_kernel);
        // INSTANTIATE(tanh_deriv_kernel);

        template activation<float> tanh(matrix<float>& in, bool derive);
        template activation<double> tanh(matrix<double>& in, bool derive);
    }
}