#ifndef Reporter_h
#define Reporter_h

#include <memory>

namespace mozart
{
    namespace reporter
    {
        class base
        {
        };

        class config
        {
        public:
            virtual std::unique_ptr<base> construct() = 0;
        };
    }
}

#endif
