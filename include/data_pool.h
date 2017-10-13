#ifndef DataPool_h
#define DataPool_h

#include <memory>
#include <unordered_map>
#include <stack>
#include <iostream>
#include <boost/compute.hpp>

using namespace std;
using namespace boost;

namespace mozart {
    template<typename T>
    class raw;

    template<typename T>
    class data_pool
    {
        friend class raw<T>;
    private:
        unordered_map<size_t, stack<std::shared_ptr<compute::vector<T>>>> _map;

        void back(std::shared_ptr<compute::vector<T>> buffer)
        {
            auto stack = this->_map.find(buffer->size());

            assert(stack != this->_map.end());

            (*stack).second.push(buffer);

            //std::cout << "BACK (" << (*stack).second.size() << " elements)" << std::endl;
        }

        std::shared_ptr<compute::vector<T>> get(size_t rows, size_t columns)
        {
            return this->get(rows, columns, context_manager::instance().context());
        }

        std::shared_ptr<compute::vector<T>> get(size_t rows, size_t columns, boost::compute::context& ctx)
        {
            size_t size = rows * columns;

            auto stack = this->_map.find(size);

            if(stack == this->_map.end())
            {
                this->_map.insert({ size, std::stack<std::shared_ptr<compute::vector<T>>>() });

                //std::cout << "NEW (stack)" << std::endl;

                return std::make_shared<compute::vector<T>>(size, ctx);
            }
            else
            {
                //std::cout << "Checking stack number of elements: " << (*stack).second.size() << std::endl;
                if((*stack).second.empty())
                {
                    //std::cout << "NEW (buffer)" << std::endl;
                    return std::make_shared<compute::vector<T>>(size, ctx);
                }
                else
                {
                    auto ret = (*stack).second.top();
                    //std::cout << "CACHE" << std::endl;
                    (*stack).second.pop();
                    return ret;
                }
            }
        }
    public:
        static data_pool& instance()
        {
            static data_pool _instance;

            return _instance;
        }

        data_pool() { }

        data_pool(const data_pool& other) = delete;
        data_pool(const data_pool&& other) = delete;

    };
}

#endif
