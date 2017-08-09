#include "function/softmax.h"

namespace mozart
{
    namespace function
    {
        inline ocl::kernel& get_softmax_kernel()
        {
            static const char * softmax_ocl_program =
            "__kernel void softmax(\n"
            "          __global float * in,\n"
            "          __global float * out,\n"
            "          __local  float * buff,\n"
            "           unsigned int isize1,\n"
            "           unsigned int isize2)\n"
            "{ \n"
            "  unsigned int gid = get_global_id(0);\n"
            "  unsigned int lid = get_local_id(0);\n"
            "  unsigned int group_size = get_local_size(0);\n"
            "  unsigned int padded = isize1 - group_size;\n"
            "  unsigned int row = gid / group_size;\n"
            "  unsigned int idx = gid + row*padded;\n"
            "  buff[lid] = exp(in[idx]);\n"
            "  barrier(CLK_LOCAL_MEM_FENCE);\n"
            "  for(int i = (group_size+1)/2; i>0; i >>= 1) {\n"
            "    if(lid < i) {\n"
            "      buff[lid] += buff[lid + i];\n"
            "    }\n"
            "    barrier(CLK_LOCAL_MEM_FENCE);\n"
            "  }\n"
            "  out[idx] = exp(in[idx]) / buff[0];\n"
            "};\n";

            auto &softmax_ocl =
                ocl::current_context().add_program(softmax_ocl_program, "softmax");
            return softmax_ocl.get_kernel("softmax");
        }

        inline ocl::kernel& get_softmax_deriv_kernel()
        {
            static const char * softmax_deriv_ocl_program =
                "__kernel void softmax_deriv(\n"
                "          __global float * in,\n"
                "          __global float * out,\n"
                "              unsigned int size1,\n"
                "              unsigned int size2,\n"
                "              unsigned int isize1)\n"
                "{ \n"
                "  unsigned int gid = get_global_id(0);\n"
                "  unsigned int gws = get_global_size(0);\n"
                "  unsigned int padded = isize1 - size1;\n"
                "  unsigned int row = gid / size2;\n"
                "  unsigned int idx = gid + row*padded;\n"
                "  out[idx] = in[idx] * (1.0f - in[idx]);\n"
                "  //out[idx] = size1;\n"
                "};\n";
            auto &softmax_deriv_ocl =
            ocl::current_context().add_program(softmax_deriv_ocl_program, "softmax_deriv");
            return softmax_deriv_ocl.get_kernel("softmax_deriv");
        }

        template<typename T>
        activation<T> softmax(matrix<T>& in, bool derive)
        {
            activation<T> result(in, derive);

            ocl::kernel &softmax_kernel = get_softmax_kernel();
            softmax_kernel.local_work_size(0, in.size2());
            softmax_kernel.global_work_size(0, in.size2() * in.size1());
            ocl::enqueue(softmax_kernel(in,
                                        result.out,
                                        local_mem(in.size2()),
                                        cl_uint(result.out.internal_size1()),
                                        cl_uint(result.out.internal_size2())));
            finish();

            if(derive) {
                ocl::kernel &softmax_deriv_kernel = get_softmax_deriv_kernel();
                softmax_deriv_kernel.local_work_size(0, in.size1());
                softmax_deriv_kernel.global_work_size(0, in.size2() * in.size1());
                ocl::enqueue(softmax_deriv_kernel(result.out,
                                                  result.deriv,
                                                  cl_uint(result.deriv.size1()),
                                                  cl_uint(result.deriv.size2()),
                                                  cl_uint(result.deriv.internal_size1())));
                finish();
            }

            return result;
        }

        template activation<float> softmax(matrix<float>& in, bool derive);
        template activation<double> softmax(matrix<double>& in, bool derive);
    }
}