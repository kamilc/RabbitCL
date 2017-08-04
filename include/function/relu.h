#ifndef ActivationReLU_h
#define ActivationReLU_h

#include <tuple>
#include "boost/optional.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/ocl/backend.hpp"
#include "utilities.h"
#include "activation.h"

using namespace std;
using namespace viennacl;
using namespace viennacl::linalg;
using namespace boost;

namespace mozart
{
    namespace function
    {
        template<typename T>
        activation<T> relu(matrix<T>& in, bool derive)
        {
            activation<T> result(in, derive);

            const char * relu_ocl_program =
            "__kernel void relu(\n"
            "          __global float * in,\n"
            "          __global float * result)\n"
            "{ \n"
            "  size_t ix = get_global_id()\n"
            "  out[ix] = (in[ix] > 0) ? in[ix] : 0;\n"
            "};\n";

            auto &relu_ocl =
                ocl::current_context().add_program(relu_ocl_program, "relu");
            ocl::kernel &relu_kernel = relu_ocl.get_kernel("relu");
            ocl::enqueue(relu_kernel(in, result.out));

            if(derive) {
                const char * relu_deriv_ocl_program =
                "__kernel void relu_deriv(\n"
                "          __global float * in,\n"
                "          __global float * result)\n"
                "{ \n"
                "  size_t ix = get_global_id()\n"
                "  out[ix] = (in[ix] > 0) ? 1 : 0;\n"
                "};\n";
                auto &relu_deriv_ocl =
                ocl::current_context().add_program(relu_deriv_ocl_program, "relu_deriv");
                ocl::kernel &relu_deriv_kernel = relu_deriv_ocl.get_kernel("relu_deriv");
                ocl::enqueue(relu_deriv_kernel(result.out, *result.deriv));
            }

            return result;
        }
    }
}

#endif