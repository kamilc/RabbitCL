#include "function/reduce_avg.h"

namespace mozart {
    namespace function {

        template<typename T>
        inline const char * reduce_avg_kernel<T>::name()
        {
            return "reduce_avg_kernel";
        }

        template<typename T>
        inline const char * reduce_avg_kernel<T>::code()
        {
            return "__kernel void reduce_avg_kernel(                                 \n"
                   "                __global float * in,                             \n"
                   "                __global float * out,                            \n"
                   "                __local  float * local_buffer,                   \n"
                   "                    unsigned int size1,                          \n"
                   "                    unsigned int size2,                          \n"
                   "                    unsigned int isize1,                         \n"
                   "                    unsigned int isize2)                         \n"
                   "{                                                                \n"
                   "    unsigned int global_id  = get_global_id(0);                  \n"
                   "    unsigned int total_size = size1 * size2;                     \n"
                   "                                                                 \n"
                   "    if(global_id < total_size)                                   \n"
                   "    {                                                            \n"
                   "      unsigned int local_id = get_local_id(0);                   \n"
                   "      unsigned int group_size = get_local_size(0);               \n"
                   "      unsigned int group_id = get_group_id(0);                   \n"
                   "      unsigned int row = global_id / size2;                      \n"
                   "      unsigned int pad = isize2 - size2;                         \n"
                   "                                                                 \n"
                   "      local_buffer[local_id] = in[global_id + row * pad];        \n"
                   "      barrier(CLK_LOCAL_MEM_FENCE);                              \n"
                   "                                                                 \n"
                   "      for(int i = ( group_size + 1 ) / 2; i > 0; i >>= 1)        \n"
                   "      {                                                          \n"
                   "        if(local_id < i)                                         \n"
                   "        {                                                        \n"
                   "          local_buffer[local_id] += local_buffer[local_id + i];  \n"
                   "        }                                                        \n"
                   "        barrier(CLK_LOCAL_MEM_FENCE);                            \n"
                   "      }                                                          \n"
                   "                                                                 \n"
                   "      if(local_id == 0)                                          \n"
                   "      {                                                          \n"
                   "        out[group_id + 1] = local_buffer[0];                     \n"
                   "      }                                                          \n"
                   "                                                                 \n"
                   "      barrier(CLK_LOCAL_MEM_FENCE);                              \n"
                   "      barrier(CLK_GLOBAL_MEM_FENCE);                             \n"
                   "                                                                 \n"
                   "      if(global_id == 0)                                         \n"
                   "      {                                                          \n"
                   "        unsigned int group_len = get_num_groups(0);              \n"
                   "                                                                 \n"
                   "        for(int i = 1; i <= group_len; i++)                      \n"
                   "        {                                                        \n"
                   "          out[0] += out[i];                                      \n"
                   "        }                                                        \n"
                   "        out[0] /= total_size;                                    \n"
                   "      }                                                          \n"
                   "    }                                                            \n"
                   "}";
        }

        template<typename T>
        void reduce_avg_kernel<T>::compute_scalar(matrix<T>& in, scalar<T>& out)
        {
            matrix<T> buffer((in.size1() * in.size2() / 2) + 1, 1);

            ocl::enqueue(this->_kernel(in,
                                       buffer,
                                       local_mem(in.size2()),
                                       cl_uint(in.size1()),
                                       cl_uint(in.size2()),
                                       cl_uint(in.internal_size1()),
                                       cl_uint(in.internal_size2())
                                       ));
            finish();

            out = buffer(0, 0);
        }

        template<typename T>
        scalar<T> reduce_avg(matrix<T>& in)
        {
            scalar<T> out(0);

            reduce_avg_kernel<T>::run_scalar(in, out, 0, 0);

            return out;
        }

        INSTANTIATE(reduce_avg_kernel);

        template scalar<float> reduce_avg(matrix<float>& in);
        template scalar<double> reduce_avg(matrix<double>& in);
    }
}