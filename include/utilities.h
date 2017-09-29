#ifndef Utilities_h
#define Utilities_h

#define INSTANTIATE(type) \
    template class type<float>; \
    template class type<double>;

#endif
