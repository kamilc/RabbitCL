#include "cost.h"

namespace mozart {

    template<typename T>
    cost<T>::cost(matrix<T> &in, bool derive)
    {
        this->out = matrix<T>(in.size1(), in.size2());
        if(derive){
            this->deriv = matrix<T>(in.size1(), in.size2());
        }
        else {
            this->deriv = matrix<T>(0, 0);
        }
    }

    template<typename T>
    T cost<T>::avg()
    {
        scalar<T> out(0.45);

        // static const char * avg_cost_ocl_program =
        // "__kernel void avg_cost(\n"
        // "          __global float * in,\n"
        // "          __global float * out,\n"
        // "              unsigned int size1,\n"
        // "              unsigned int size2,\n"
        // "              unsigned int isize1)\n"
        // "{ \n"
        // "  unsigned int gid = get_global_id(0);\n"
        // "  unsigned int padded = isize1 - size1;\n"
        // "  unsigned int row = gid / size2;\n"
        // "  unsigned int idx = gid + row*padded;\n"
        // "  out[idx] = fmax(0.0f, in[idx]);\n"
        // "};\n";

        // auto &avg_cost_ocl =
        //     ocl::current_context().add_program(avg_cost_ocl_program, "avg_cost");
        // ocl::kernel &avg_cost_kernel = avg_cost_ocl.get_kernel("avg_cost");

        // avg_cost_kernel.local_work_size(0, this->out.size1());
        // avg_cost_kernel.global_work_size(0, this->out.size2() * this->out.size1());
        // ocl::enqueue(relu_kernel(in,
        //                          out,
        //                          cl_uint(this->out.size1()),
        //                          cl_uint(this->out.size2()),
        //                          cl_uint(this->out.internal_size1())));
        // finish();

        return out;
    }

    INSTANTIATE(cost);
}