#include "function/tanh.h"

namespace mozart
{
    namespace function
    {
        // todo: provide the activation function also for the
        // speedy native_ variants
        inline ocl::kernel& get_tanh_kernel()
        {
            static const char * tanh_ocl_program =
            "__kernel void tanh_activation(\n"
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
            "  out[idx] = tanh(in[idx]);\n"
            "};\n";

            auto &tanh_ocl =
                ocl::current_context().add_program(tanh_ocl_program, "tanh_activation");
            return tanh_ocl.get_kernel("tanh_activation");
        }

        inline ocl::kernel& get_tanh_deriv_kernel()
        {
            static const char * tanh_deriv_ocl_program =
                "__kernel void tanh_deriv(\n"
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
                "  out[idx] = 1.0 - in[idx] * in[idx];\n"
                "};\n";
            auto &tanh_deriv_ocl =
            ocl::current_context().add_program(tanh_deriv_ocl_program, "tanh_deriv");
            return tanh_deriv_ocl.get_kernel("tanh_deriv");
        }

        template<typename T>
        activation<T> tanh(matrix<T>& in, bool derive)
        {
            activation<T> result(in, derive);

            ocl::kernel &tanh_kernel = get_tanh_kernel();
            tanh_kernel.local_work_size(0, in.size1());
            tanh_kernel.global_work_size(0, in.size2() * in.size1());
            ocl::enqueue(tanh_kernel(in,
                                     result.out,
                                     cl_uint(result.out.size1()),
                                     cl_uint(result.out.size2()),
                                     cl_uint(result.out.internal_size1())));
            finish();

            if(derive) {
                ocl::kernel &tanh_deriv_kernel = get_tanh_deriv_kernel();
                tanh_deriv_kernel.local_work_size(0, in.size1());
                tanh_deriv_kernel.global_work_size(0, in.size2() * in.size1());
                ocl::enqueue(tanh_deriv_kernel(result.out,
                                               result.deriv,
                                               cl_uint(result.deriv.size1()),
                                               cl_uint(result.deriv.size2()),
                                               cl_uint(result.deriv.internal_size1())));
                finish();
            }

            return result;
        }

        template activation<float> tanh(matrix<float>& in, bool derive);
        template activation<double> tanh(matrix<double>& in, bool derive);
    }
}