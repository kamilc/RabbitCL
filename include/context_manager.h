#ifndef ContextManager_h
#define ContextManager_h

#include <boost/compute/core.hpp>
#include <cstdlib>
#include <string>

using namespace boost;

namespace mozart
{
    class context_manager
    {
    private:
        bool _initialized = false;
    public:
        static context_manager& instance()
        {
            static context_manager _instance;
            if(!_instance._initialized)
            {
                _instance._context = compute::context(_instance.device());
                _instance._initialized = true;
            }
            return _instance;
        }

        context_manager() {}
        context_manager(context_manager const&) = delete;
        void operator=(context_manager const&)  = delete;

        compute::context& context()
        {
            return this->_context;
        }

        compute::device device()
        {
            auto string_number = std::getenv("OPENCL_DEVICE");
            int number = std::stoi(string_number == nullptr ? std::string{ "0" } : std::string{string_number});

            auto device = boost::compute::system::devices()[number];

            return device;
        }

        compute::command_queue new_queue()
        {
            return compute::command_queue(this->context(), this->device());
        }
    private:
        compute::context _context;
    };
}

#endif
