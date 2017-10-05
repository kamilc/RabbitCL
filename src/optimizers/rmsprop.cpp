#include "optimizers/rmsprop.h"

namespace mozart
{
    namespace optimizers
    {
        template<typename T>
        rmsprop<T>::rmsprop(typename cost<T>::function func) : gradient_descent<T>(func)
        {
            // no-op
        }

        template<typename T>
        rmsprop<T>& rmsprop<T>::alpha(T alpha)
        {
            this->_alpha = alpha;

            return *this;
        }

        template<typename T>
        rmsprop<T>& rmsprop<T>::mu(T mu)
        {
            this->_mu = mu;

            return *this;
        }

        template<typename T>
        rmsprop<T>& rmsprop<T>::eps(T eps)
        {
            this->_eps = eps;

            return *this;
        }

        template<typename T>
        matrix<T>& rmsprop<T>::memo_for_index(size_t index, matrix<T>& deltas)
        {
            if(this->_memo.find(index) == this->_memo.end())
            {
                this->_memo[index] = std::make_shared<matrix<T>>(deltas.size1(), deltas.size2());
            }

            return *this->_memo[index];
        }

        template<typename T>
        void rmsprop<T>::update(size_t index, std::shared_ptr<layer<T>> layer, matrix<T>& delta, matrix<T>& weight_delta)
        {
            matrix<T>& memo = this->memo_for_index(index, weight_delta);

            rmsprop_update(this->_alpha, weight_delta, memo, this->_eps, this->_mu);

            layer->update_bias(delta);
            layer->update_weights(weight_delta);
        }

        INSTANTIATE(rmsprop);
    }
}
