#ifndef Timed_h
#define Timed_h

#include <memory>
#include <chrono>
#include "utilities.h"
#include "stat.h"
#include "reporter.h"
#include "cost.h"

using namespace std;

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
            timed(std::chrono::duration<int>);

            timed& stats(typename stat<T>::function);
            timed& epoch_timing(bool);

            std::unique_ptr<base<T>> construct();
        private:
            std::chrono::duration<int> _interval;
            typename stat<T>::function _function;
            bool _epoch_timing;
        };

        template<typename T>
        class timed_reporter : public base<T>
        {
        public:
            timed_reporter(timed<T>& config);
            void push_error(cost<T>&);
            void start_epoch(unsigned int, unsigned int);
        private:
            std::chrono::duration<int> _interval;
            std::chrono::system_clock::time_point _last_epoch_start;
            typename stat<T>::function _function;
            bool _epoch_timing;
            T _last_error;
        };
    }
}

#endif
