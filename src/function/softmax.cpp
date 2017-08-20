#include "function/softmax.h"

namespace mozart
{
    namespace function
    {
        
        // template<typename T>
        // inline const char * softmax_kernel<T>::name()
        // {
        //     return "softmax_kernel";
        // }

        // template<typename T>
        // inline const char * softmax_kernel<T>::code()
        // {
        //     return  "__kernel void softmax_kernel(\n"
        //             "          __global float * in,\n"
        //             "          __global float * out,\n"
        //             "          __local  float * buff,\n"
        //             "           unsigned int isize1,\n"
        //             "           unsigned int isize2)\n"
        //             "{ \n"
        //             "  unsigned int gid = get_global_id(0);\n"
        //             "  unsigned int lid = get_local_id(0);\n"
        //             "  unsigned int group_size = get_local_size(0);\n"
        //             "  unsigned int padded = isize1 - group_size;\n"
        //             "  unsigned int row = gid / group_size;\n"
        //             "  unsigned int idx = gid + row*padded;\n"
        //             "  buff[lid] = exp(in[idx]);\n"
        //             "  barrier(CLK_LOCAL_MEM_FENCE);\n"
        //             "  for(int i = (group_size+1)/2; i>0; i >>= 1) {\n"
        //             "    if(lid < i) {\n"
        //             "      buff[lid] += buff[lid + i];\n"
        //             "    }\n"
        //             "    barrier(CLK_LOCAL_MEM_FENCE);\n"
        //             "  }\n"
        //             "  out[idx] = exp(in[idx]) / buff[0];\n"
        //             "};\n";
        // }

        // template<typename T>
        // void softmax_kernel<T>::compute_matrix(matrix<T>& in, matrix<T>& out)
        // {
        //     ocl::enqueue(this->_kernel( in,
        //                                 out,
        //                                 local_mem(in.size2()),
        //                                 cl_uint(out.internal_size1()),
        //                                 cl_uint(out.internal_size2())));
        //     finish();
        // }

        // template<typename T>
        // inline const char * softmax_deriv_kernel<T>::name()
        // {
        //     return "softmax_deriv_kernel";
        // }

        // template<typename T>
        // inline const char * softmax_deriv_kernel<T>::code()
        // {
        //     return  "__kernel void softmax_deriv_kernel(\n"
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
        //             "  out[idx] = in[idx] * (1.0f - in[idx]);\n"
        //             "};\n";
        // }

        // template<typename T>
        // void softmax_deriv_kernel<T>::compute_matrix(matrix<T>& in, matrix<T>& out)
        // {
        //     ocl::enqueue(this->_kernel( in,
        //                                 out,
        //                                 cl_uint(out.size1()),
        //                                 cl_uint(out.size2()),
        //                                 cl_uint(out.internal_size1())));
        //     finish();
        // }

        template<typename T>
        activation<T> softmax(matrix<T>& in, bool derive)
        {
            activation<T> result(in, derive);

            //softmax_kernel<T>::run_matrix(in, result.out, in.size1(), in.size2() * in.size1());

            if(derive) {
                //softmax_deriv_kernel<T>::run_matrix(result.out, result.deriv, in.size1(), in.size2() * in.size1());
            }

            return result;
        }

        // INSTANTIATE(softmax_kernel);
        // INSTANTIATE(softmax_deriv_kernel);

        template activation<float> softmax(matrix<float>& in, bool derive);
        template activation<double> softmax(matrix<double>& in, bool derive);
    }
}