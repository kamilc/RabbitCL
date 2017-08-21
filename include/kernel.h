#ifndef KernelClass_h
#define KernelClass_h

#include <cstddef>
#include <boost/compute/kernel.hpp>
#include <boost/compute/command_queue.hpp>
#include "matrix_size.h"

using namespace std;
using namespace boost;

#define KERNEL_NAME(name) \
struct name \
{ \
    static const char* to_s() \
    { \
        return "#name"; \
    } \
}; \

namespace mozart
{
    template<typename T>
    class kernel_base
    {
    protected:
        compute::kernel _kernel;
        compute::command_queue _queue;
        bool _compiled = false;
        size_t _position;

    public:
        template<typename A1, typename... Args>
        void run_(A1 arg, Args ...args)
        {
            this->_kernel.set_arg(this->_position++, arg);
            run(args...);
        }

        template<typename A1>
        void run_(A1 arg)
        {
            this->_kernel.set_arg(this->_position++, arg);
        }

        template<typename... Args>
        void run(Args ...args)
        {
            this->_position = 0;
            run_(args...);
        }

        void compile()
        {
            if(!this->_compiled)
            {
                //auto &program = ocl::current_context().add_program(this->code(), NAME::to_s());
                //this->_kernel = program.get_kernel(NAME::to_s());
                //this->_compiled = true;
            }
        }
    };

    template<typename T, typename NAME>
    class kernel : public kernel_base<T>
    {
    public:
        const char * code();

        kernel() { }
    };
}

#define KERNEL(name, source) \
    KERNEL_NAME(name); \
    template<typename T> \
    class kernel<T, name> : public kernel_base<T> \
    { \
    public: \
        const char * code() \
        { \
            return BOOST_COMPUTE_STRINGIZE_SOURCE( \
                source \
            ); \
        } \
        static kernel& instance() \
        { \
            static kernel _instance; \
            _instance.compile(); \
            return _instance; \
        } \
    }; \

#endif