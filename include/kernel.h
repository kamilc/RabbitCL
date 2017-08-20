#ifndef KernelClass_h
#define KernelClass_h

#define KERNEL(name) \
struct name \
{ \
    static const char* to_s() \
    { \
        return "#name"; \
    } \
}; \

template<typename T, typename NAME>
class kernel
{
public:
    static kernel& instance()
    {
        static kernel _instance;
        _instance.compile();
        return _instance;
    }

    kernel() { }

    void compile()
    {
        if(!this->_compiled)
        {
            // auto &program = ocl::current_context().add_program(this->code(), NAME::to_s());
            // this->_kernel = program.get_kernel(NAME::to_s());
            // this->_compiled = true;
        }
    }

    template<typename T1>
    void operator()(T1 const &t1)
    {
        // auto _kernel = this(t1);
        // ocl::enqueue(_kernel);
        // finish();
    }
    
    template<typename T1, typename T2>
    void operator()(T1 const &t1, T2 const &t2)
    {
        // auto _kernel = this(t1, t2);
        // ocl::enqueue(_kernel);
        // finish();
    }
    
    template<typename T1, typename T2, typename T3>
    void operator()(T1 const &t1, T2 const &t2, T3 const &t3)
    {
        // auto _kernel = this(t1, t2, t3);
        // ocl::enqueue(_kernel);
        // finish();
    }
    
    template<typename T1, typename T2, typename T3, typename T4>
    void operator()(T1 const &t1, T2 const &t2, T3 const &t3, T4 const &t4)
    {
        // auto _kernel = this(t1, t2, t3, t4);
        // ocl::enqueue(_kernel);
        // finish();
    }
    
    template<typename T1, typename T2, typename T3, typename T4, typename T5>
    void operator()(T1 const &t1, T2 const &t2, T3 const &t3, T4 const &t4, T5 const &t5)
    {
        // auto _kernel = this(t1, t2, t3, t4, t5);
        // ocl::enqueue(_kernel);
        // finish();
    }
private:
    // ocl::kernel _kernel;
    bool _compiled = false; 

    inline const char * code();
};

#endif