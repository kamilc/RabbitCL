#include "function/adagrad_update.h"

namespace mozart
{
    KERNEL(adagrad_update_kernel,
        __kernel void adagrad_update_kernel(
                    const TYPE alpha,
                    const TYPE eps,
                    __global TYPE * weight_delta,
                    __global TYPE * memo,
                   const struct matrix_size size)
        {
            unsigned int global_id = get_global_id(0);
            unsigned int idx = id_to_internal_id(global_id, &size);

            // taken from:
            // https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/optimizers/optimizer.h
            //
            // g[i] += dW[i] * dW[i];
            // W[i] -= alpha * dW[i] / (std::sqrt(g[i]) + eps);

            TYPE weight_delta_value = weight_delta[idx];
            TYPE new_memo_value = memo[idx] + pow(weight_delta_value, 2);

            weight_delta[idx] = -1 * alpha * weight_delta_value / ( sqrt(new_memo_value) + eps );
            memo[idx] = new_memo_value;
        };
    );

    namespace function
    {
        template<typename T>
        void adagrad_update(T alpha, matrix<T>& weight_delta, matrix<T>& memo, T eps)
        {
          //std::cout << "Weight Delta" << weight_delta << "Memo: " << memo << "Alpha " << alpha << " Eps: " << eps << std::endl;
            kernel<T, adagrad_update_kernel>::instance()
                .with_global_size(weight_delta.total_size())
                // todo: infer as large of a localsize as possible
                .with_local_size(weight_delta.size1())
                .run(
                    alpha,
                    eps,
                    weight_delta,
                    memo,
                    weight_delta.size()
                );
        }

        template void adagrad_update(float alpha, matrix<float>& weight_delta, matrix<float>& memo, float eps);
        template void adagrad_update(double alpha, matrix<double>& weight_delta, matrix<double>& memo, double eps);
    }
}
