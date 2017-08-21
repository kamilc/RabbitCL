#ifndef KernelClass_h
#define KernelClass_h

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION 1

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
        return BOOST_COMPUTE_STRINGIZE_SOURCE(name); \
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
        const char * _name;

    public:
        template<typename A1, typename... Args>
        void run_(A1 arg, Args ...args)
        {
            this->_kernel.set_arg(this->_position++, arg);
            run(args...);
        }

        template<typename... Args>
        void run_(matrix_size arg, Args ...args)
        {
            this->_kernel.set_arg(this->_position++, sizeof(arg), &arg);
            run(args...);
        }

        void run_(matrix_size arg)
        {
            this->_kernel.set_arg(this->_position++, sizeof(arg), &arg);
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

        compute::context context()
        {
            // todo: implement me
            return compute::context(compute::system::default_device());
        }

        virtual const char * code() = 0;

        void compile()
        {
            if(!this->_compiled)
            {
                compute::program _program =
                    compute::program::create_with_source(this->code(), this->context());
                try {
                    _program.build();
                    this->_kernel = compute::kernel(_program, this->_name);
                    this->_compiled = true;
                }
                catch(boost::compute::opencl_error &e){
                    std::cout << e.error_string() << std::endl;
                    std::cout << _program.build_log() << std::endl;
                }
                
            }
        }
    };

    template<typename T, typename NAME>
    class kernel : public kernel_base<T>
    {
    public:
        const char * code();

        kernel() {
            std::cout << "Setting name to " << NAME::to_s() << std::endl;
            this->_name = NAME::to_s();
        }
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
                struct matrix_size \
                { \
                    unsigned int size1; \
                    unsigned int size2; \
                    unsigned int start1; \
                    unsigned int start2; \
                    unsigned int internal_size1; \
                    unsigned int internal_size2; \
                    unsigned int transposed; \
                }; \
                source \
            ); \
        } \
        kernel<T, name>() { \
            std::cout << "Setting name to " << name::to_s() << std::endl; \
            this->_name = name::to_s(); \
        } \
        static kernel& instance() \
        { \
            static kernel _instance; \
            _instance.compile(); \
            return _instance; \
        } \
    }; \

#endif