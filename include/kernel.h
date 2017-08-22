#ifndef KernelClass_h
#define KernelClass_h

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION 1

#include <cstddef>
#include <boost/compute/kernel.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/buffer.hpp>
#include "matrix.h"
#include "scalar.h"
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
            run_(args...);
        }

        template<typename... Args>
        void run_(matrix_size arg, Args ...args)
        {
            this->_kernel.set_arg(this->_position++, sizeof(arg), &arg);
            run_(args...);
        }

        void run_(matrix_size arg)
        {
            this->_kernel.set_arg(this->_position++, sizeof(arg), &arg);
            this->run();
        }

        template<typename... Args>
        void run_(matrix<T>& arg, Args ...args)
        {
            this->_kernel.set_arg(this->_position++, arg.data());
            run_(args...);
        }

        void run_(matrix<T> &arg)
        {
            this->_kernel.set_arg(this->_position++, arg.data());
            this->run();
        }

        template<typename... Args>
        void run_(scalar<T>& arg, Args ...args)
        {
            this->_kernel.set_arg(this->_position++, arg.data());
            run_(args...);
        }

        void run_(scalar<T> &arg)
        {
            this->_kernel.set_arg(this->_position++, arg.data());
            this->run();
        }
        
        template<typename A1>
        void run_(A1 arg)
        {
            this->_kernel.set_arg(this->_position++, arg);
            this->run();
        }

        template<typename... Args>
        void run(Args ...args)
        {
            this->_position = 0;
            compute::system::default_queue().finish();
            run_(args...);
        }

        void run()
        {
            std::cout << "run()" << std::endl;

            auto queue = context_manager::instance().new_queue();

            std::cout << "About to run with " << this->_position << " arguments" << std::endl;

            // todo: implement me properly
            queue.enqueue_1d_range_kernel(
                this->_kernel,
                0,
                16,
                16
            );

            std::cout << "Enqueued" << std::endl;

            queue.finish();
        }

        virtual const char * code() = 0;

        void compile()
        {
            if(!this->_compiled)
            {
                compute::program _program =
                    compute::program::create_with_source(this->code(), context_manager::instance().context());
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