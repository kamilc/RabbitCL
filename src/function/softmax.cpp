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
            "           size_t isize1,\n"
            "           size_t isize2)\n"
            "{ \n"
            "  size_t gid = get_global_id(0);\n"
            "  size_t lid = get_local_id(0);\n"
            "  size_t group_size = get_local_size(0);\n"
            "  size_t padded = isize1 - group_size;\n"
            "  size_t row = gid / group_size;\n"
            "  size_t idx = gid + row*padded;\n"
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
                "          __global float * out)\n"
                "{ \n"
                "  size_t ix = get_global_id(0);\n"
                "  out[ix] = (in[ix] > 0) ? 1 : 0;\n"
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

            if(derive) {
                
                ocl::kernel &softmax_deriv_kernel = get_softmax_deriv_kernel();
                ocl::enqueue(softmax_deriv_kernel(result.out, *result.deriv));
            }

            return result;
        }

        template activation<float> softmax(matrix<float>& in, bool derive);
        template activation<double> softmax(matrix<double>& in, bool derive);
    }
}