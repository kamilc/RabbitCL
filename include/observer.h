#ifndef Reporter_h
#define Reporter_h

#include <memory>
#include "matrix.h"
#include "cost.h"
#include "activation.h"
#include "sequence.h"

namespace mozart
{
    namespace observer
    {
        template<typename T>
        class base
        {
        public:
            virtual void push_outputs(activation<T>&, matrix<T>&)
            {
                // no-op
            }

            virtual void push_error(cost<T>&)
            {
                // no-op
            }

            virtual void start()
            {
                // no-op
            }

            virtual void start_batch(unsigned int, unsigned int)
            {
                // no-op
            }

            virtual void start_epoch(unsigned int, unsigned int)
            {
                // no-op
            }

            virtual void end()
            {
                // no-op
            }

            virtual void end_batch()
            {
                // no-op
            }

            virtual void end_epoch(sequence<T>&)
            {
                // no-op
            }
        };

        template<typename T>
        class config
        {
        public:
            virtual std::unique_ptr<base<T>> construct() = 0;
        };
    }
}

#endif
