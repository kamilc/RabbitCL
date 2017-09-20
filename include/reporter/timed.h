#ifndef Timed_h
#define Timed_h

#include <memory>
#include "utilities.h"
#include "stat.h"
#include "reporter.h"

namespace mozart
{
    namespace reporter
    {
        template<typename T>
        class timed_reporter;

        template<typename T>
        class timed : public config<T>
        {
          friend class timed_reporter<T>;
        public:
            timed(double ms);

            timed& stats(typename stat<T>::function);
            timed& epoch_timing(bool);

            std::unique_ptr<base<T>> construct();
        private:
            double _interval;
            typename stat<T>::function _function;
            bool _epoch_timing;
        };

        template<typename T>
        class timed_reporter : public base<T>
        {
        public:
            timed_reporter(timed<T>& config);
        private:
            double _interval;
            typename stat<T>::function _function;
            bool _epoch_timing;
        };
    }
}

#endif
