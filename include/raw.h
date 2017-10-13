#ifndef Raw_h
#define Raw_h

#include <memory>
#include <boost/compute.hpp>

#include "data_pool.h"

using namespace std;
using namespace boost;

namespace mozart {
    template<typename T>
    class raw
    {
    public:
        raw(size_t rows, size_t cols)
        {
            this->_data = data_pool<T>::instance().get(rows, cols);
        }

        raw(size_t rows, size_t cols, boost::compute::context &ctx)
        {
            this->_data = data_pool<T>::instance().get(rows, cols, ctx);
        }

        ~raw()
        {
            data_pool<T>::instance().back(this->_data);
        }

        std::shared_ptr<compute::vector<T>>& operator*()
        {
            return this->_data;
        }
    private:
        std::shared_ptr<compute::vector<T>> _data;
    };
}

#endif
