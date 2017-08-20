#ifndef Scalar_h
#define Scalar_h

namespace mozart
{
    template<typename T>
    class scalar
    {
    public:
        scalar();
        scalar(T initial);

        void operator=(T value);
        operator T();
    };
}

#endif