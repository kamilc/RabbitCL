#ifndef KernelClass_h
#define KernelClass_h

#define KERNEL_CLASS(kernel_name) \
template<typename T> \
class kernel_name \
{ \
public: \
    static kernel_name& instance() \
    { \
        static kernel_name _instance;\
        _instance.compile();\
        return _instance;\
    }\
    kernel_name() { }\
    void compile()\
    {\
        if(!this->_compiled) \
        { \
            auto &program = ocl::current_context().add_program(this->code(), this->name());\
            this->_kernel = program.get_kernel(this->name());\
            this->_compiled = true; \
        } \
    }\
    void compute_scalar(matrix<T>& in, scalar<T>& out);\
    void compute_matrix(matrix<T>& in, matrix<T>& out);\
    static void run_matrix(matrix<T>& in, matrix<T>& out, std::size_t local_size, std::size_t global_size) \
    { \
        kernel_name kernel = kernel_name::instance(); \
        kernel.local_work_size(0, local_size); \
        kernel.global_work_size(0, global_size); \
        kernel.compute_matrix(in, out); \
    } \
    void local_work_size(std::size_t dimension, std::size_t size) { \
        this->_kernel.local_work_size(dimension, size); \
    }\
    void global_work_size(std::size_t dimension, std::size_t size) { \
        this->_kernel.global_work_size(dimension, size); \
    }\
private:\
    inline const char * name();\
    inline const char * code();\
    ocl::kernel _kernel;\
    bool _compiled = false; \
}\

#endif