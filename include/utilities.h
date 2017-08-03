#ifndef Utilities_h
#define Utilities_h

#define INSTANTIATE(type) \
    template class type<float, mode::cpu>; \
    template class type<float, mode::gpu>; \
    template class type<double, mode::cpu>; \
    template class type<double, mode::gpu>; \

#endif