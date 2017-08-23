#include "function/reduce_avg.h"

namespace mozart {
    KERNEL(reduce_avg_kernel,
        __kernel void reduce_avg_kernel(              
                    __global TYPE * in,
                    __global TYPE * out,
                     __local TYPE * local_buffer,
                  struct matrix_size in_size)
        {
            unsigned int global_id  = get_global_id(0);
            unsigned int total_size = in_size.size1 * in_size.size2;

            printf("total size is: %d\n", total_size);

            if(global_id < total_size)                
            {                                         
                unsigned int local_id = get_local_id(0);
                unsigned int group_id = get_group_id(0);
                unsigned int group_size = get_global_size(0) / get_local_size(0);
                unsigned int row = global_id / in_size.size2;   
                unsigned int pad = in_size.internal_size2 - in_size.size2;

                printf("local_id = %d | group_size = %d | group_id = %d | row = %d | pad = %d\n",
                    local_id, group_size, group_id, row, pad);
                                                    
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

            kernel<T, reduce_avg_kernel>::instance()
                .with_global_size(in.total_size())
                .with_local_size(in.size1())
                .run(in,
                    out,
                    // todo: figure out best dim for local buffer here:
                    local<T>(in.total_size()),
                    in.size());

            return out;
        }

        template scalar<float> reduce_avg(matrix<float>& in);
        template scalar<double> reduce_avg(matrix<double>& in);
    }
}