#include "function/squared_error.h"

namespace mozart
{
    namespace function
    {
        // todo: implement me
        inline ocl::kernel& get_relu_kernel()
        {

            static const char * relu_ocl_program =
            "__kernel void relu(\n"
            "          __global float * in,\n"
            "          __global float * out,\n"
            "              unsigned int size1,\n"
            "              unsigned int size2,\n"
            "              unsigned int isize1)\n"
            "{ \n"
            "  unsigned int gid = get_global_id(0);\n"
            "  unsigned int padded = isize1 - size1;\n"
            "  unsigned int row = gid / size2;\n"
            "  unsigned int idx = gid + row*padded;\n"
            "  out[idx] = fmax(0.0f, in[idx]);\n"
            "};\n";

            auto &relu_ocl =
                ocl::current_context().add_program(relu_ocl_program, "relu");
            return relu_ocl.get_kernel("relu");
        }

        inline ocl::kernel& get_relu_deriv_kernel()
        {
            static const char * relu_deriv_ocl_program =
                "__kernel void relu_deriv(\n"
                "          __global float * in,\n"
                "          __global float * out,\n"
                "              unsigned int size1,\n"
                "              unsigned int size2,\n"
                "              unsigned int isize1)\n"
                "{ \n"
                "  unsigned int gid = get_global_id(0);\n"
                "  unsigned int padded = isize1 - size1;\n"
                "  unsigned int row = gid / size2;\n"
                "  unsigned int idx = gid + row*padded;\n"
                "  out[idx] = fmin(1.0f, floor(in[idx]));\n"
                "};\n";
            auto &relu_deriv_ocl =
            ocl::current_context().add_program(relu_deriv_ocl_program, "relu_deriv");
            return relu_deriv_ocl.get_kernel("relu_deriv");
        }

        template<typename T>
        cost<T> squared_error(matrix<T>& in, bool derive)
        {
            cost<T> result(in, derive);

            ocl::kernel &relu_kernel = get_relu_kernel();
            relu_kernel.local_work_size(0, in.size1());
            relu_kernel.global_work_size(0, in.size2() * in.size1());
            ocl::enqueue(relu_kernel(in,
                                     result.out,
                                     cl_uint(result.out.size1()),
                                     cl_uint(result.out.size2()),
                                     cl_uint(result.out.internal_size1())));
            finish();

            if(derive) {
                ocl::kernel &relu_deriv_kernel = get_relu_deriv_kernel();
                relu_deriv_kernel.local_work_size(0, in.size1());
                relu_deriv_kernel.global_work_size(0, in.size2() * in.size1());
                ocl::enqueue(relu_deriv_kernel(result.out,
                                               result.deriv,
                                               cl_uint(result.deriv.size1()),
                                               cl_uint(result.deriv.size2()),
                                               cl_uint(result.deriv.internal_size1())));
                finish();
            }

            return result;
        }

        template cost<float> squared_error(matrix<float>& in, bool derive);
        template cost<double> squared_error(matrix<double>& in, bool derive);
    }
}