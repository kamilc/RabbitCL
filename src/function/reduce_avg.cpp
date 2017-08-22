#include "function/reduce_avg.h"

namespace mozart {
    KERNEL(reduce_avg_kernel,
        __kernel void reduce_avg_kernel(              
                    __global float * in,
                    __global float * out,
                  struct matrix_size in_size)
        {
            __local float * local_buffer;

            unsigned int global_id  = get_global_id(0);
            unsigned int total_size = in_size.size1 * in_size.size2;

            if(global_id < total_size)                
            {                                         
                unsigned int local_id = get_local_id(0);
                unsigned int group_size = get_local_size(0);       
                unsigned int group_id = get_group_id(0);
                unsigned int row = global_id / in_size.size2;   
                unsigned int pad = in_size.internal_size2 - in_size.size2;      
                                                    
                local_buffer[local_id] = in[global_id + row * pad];
                barrier(CLK_LOCAL_MEM_FENCE);           
                                                    
                for(int i = ( group_size + 1 ) / 2; i > 0; i >>= 1)
                {                                       
                    if(local_id < i)                      
                    {                                     
                    local_buffer[local_id] += local_buffer[local_id + i];
                    }                                     
                    barrier(CLK_LOCAL_MEM_FENCE);         
                }                                       
                                                    
                if(local_id == 0)                       
                {                                       
                    out[group_id + 1] = local_buffer[0];  
                }                                       
                                                    
                barrier(CLK_LOCAL_MEM_FENCE);           
                barrier(CLK_GLOBAL_MEM_FENCE);          
                                                    
                if(global_id == 0)                      
                {                                       
                    unsigned int group_len = get_num_groups(0);      
                                                        
                    for(int i = 1; i <= group_len; i++)   
                    {                                     
                    out[0] += out[i];                   
                    }
                    out[0] /= total_size;
                }
            }                                         
        }
    )

    namespace function {

        template<typename T>
        scalar<T> reduce_avg(matrix<T>& in)
        {
            scalar<T> out(0);

            kernel<T, reduce_avg_kernel>::instance().run(in, out, in.size());

            return out;
        }

        template scalar<float> reduce_avg(matrix<float>& in);
        template scalar<double> reduce_avg(matrix<double>& in);
    }
}