#ifndef ContextManager_h
#define ContextManager_h

#include <boost/compute/core.hpp>

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
                _instance._context = compute::context(compute::system::default_device());
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
            return compute::system::default_device();
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