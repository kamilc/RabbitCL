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
                _instance._context = compute::context(boost::compute::system::devices()[2]);
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
            // todo: make it choosable:

            auto device = boost::compute::system::devices()[2];

            //std::cout << "Choosing " << device.name() << std::endl;

            //return boost::compute::system::find_device("AMD Radeon Pro 455 Compute Engine");

            //return compute::system::default_device();

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