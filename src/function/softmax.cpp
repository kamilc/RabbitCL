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
            "          __global float * out)\n"
            "{ \n"
            "  size_t ix = get_global_id(0);\n"
            "  out[ix] = (in[ix] > 0) ? in[ix] : 0;\n"
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
            ocl::enqueue(softmax_kernel(in, result.out));

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