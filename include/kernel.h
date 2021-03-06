#ifndef KernelClass_h
#define KernelClass_h

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION 1

#include <cstddef>
#include <boost/compute/kernel.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/buffer.hpp>
#include <string>
#include <regex>
#include "matrix_size.h"
#include "local.h"

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
    class matrix;

    template<typename T>
    class scalar;

    template<typename T>
    class kernel_base
    {
    protected:
        compute::kernel _kernel;
        compute::program _program;
        compute::command_queue _queue;
        bool _compiled = false;
        size_t _position = 0;
        size_t _global_work_size = 0;
        size_t _local_work_size = 0;
        const char * _name;

        void clean()
        {
            this->_position = 0;
            this->_global_work_size = 0;
            this->_local_work_size = 0;
        }

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

        void run_(local<T> arg)
        {
            this->_kernel.set_arg(this->_position++, arg.memory_size(), NULL);
            this->run();
        }

        template<typename... Args>
        void run_(local<T> arg, Args ...args)
        {
            this->_kernel.set_arg(this->_position++, arg.memory_size(), NULL);
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
            run_(args...);
        }

        void run()
        {
            auto queue = context_manager::instance().new_queue();

            queue.finish();

            if(this->_global_work_size == 0 || this->_local_work_size == 0)
            {
                this->infer_work_sizes();
            }

            auto _event = queue.enqueue_1d_range_kernel(
                this->_kernel,
                0,
                this->_global_work_size,
                this->_local_work_size
            );

            _event.wait();
            queue.finish();
        }

        void infer_work_sizes()
        {
            // todo: implement me in a smart way
            this->_global_work_size = 128;
            this->_local_work_size = 128;
        }

        kernel_base& with_global_size(size_t size)
        {
            this->_global_work_size = size;

            return *this;
        }

        kernel_base& with_local_size(size_t size)
        {
            this->_local_work_size = size;

            return *this;
        }

        std::string type_string(float param)
        {
            return "float";
        }

        std::string type_string(double param)
        {
            return "double";
        }

        virtual const char * code() = 0;

        void compile()
        {
            if(!this->_compiled)
            {
                std::regex type_regex("TYPE");
                std::regex pragma_regex("@pragma");
                std::regex kernel_name_regex(this->_name);
                std::string _processed_name =
                    std::string(this->_name) + "_" + this->type_string(T{0});

                std::string _processed_code = std::regex_replace(
                    this->code(),
                    type_regex,
                    this->type_string(T{0})
                );

                _processed_code = std::regex_replace(
                    _processed_code,
                    kernel_name_regex,
                    _processed_name
                );

                _processed_code = std::regex_replace(
                    _processed_code,
                    pragma_regex,
                    std::string("\n\n#pragma")
                );

                _processed_code = std::string(BOOST_COMPUTE_STRINGIZE_SOURCE(
                    struct matrix_size
                    {
                        unsigned int size1;
                        unsigned int size2;
                        unsigned int start1;
                        unsigned int start2;
                        unsigned int internal_size1;
                        unsigned int internal_size2;
                    };

                    inline unsigned int id_to_internal_id(
                        unsigned int global_id,
                        struct matrix_size * size)
                    {
                        unsigned int row = global_id / size->size2;
                        unsigned int pad = size->internal_size2 - size->size2;

                        return global_id + row * pad +
                            size->start1 * size->internal_size2 +
                            size->start2;
                    }
                )) + _processed_code;

                #ifdef MOZART_DEBUG
                std::cout << _processed_name << std::endl;
                std::cout << _processed_code << std::endl;
                #endif

                this->_program =
                    compute::program::create_with_source(
                        _processed_code,
                        context_manager::instance().context()
                    );
                try {
                    this->_program.build();
                    this->_kernel = compute::kernel(this->_program, _processed_name);
                    this->_compiled = true;
                }
                catch(boost::compute::opencl_error &e){
                    std::cout << e.error_string() << std::endl;
                    std::cout << this->_program.build_log() << std::endl;
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
                source \
            ); \
        } \
        kernel<T, name>() { \
            this->_name = name::to_s(); \
        } \
        static kernel& instance() \
        { \
            static kernel _instance; \
            _instance.clean(); \
            _instance.compile(); \
            return _instance; \
        } \
    }; \

#endif
