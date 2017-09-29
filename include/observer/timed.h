#ifndef Timed_h
#define Timed_h

#include <memory>
#include <chrono>
#include <thread>
#include "utilities.h"
#include "stat.h"
#include "observer.h"
#include "cost.h"

using namespace std;

namespace mozart
{
    namespace observer
    {
        template<typename T>
        class timed_observer;

        template<typename T>
        class timed : public config<T>
        {
          friend class timed_observer<T>;
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
        class timed_observer : public base<T>
        {
        public:
            timed_observer(timed<T>& config);
            void push_error(cost<T>&);
            void start_epoch(unsigned int, unsigned int);
            void end_epoch(sequence<T>&);
            void start();
            void end();
        private:
            void main();

            std::chrono::duration<int> _interval;
            std::chrono::duration<double> _last_epoch_timing;
            std::chrono::system_clock::time_point _last_epoch_start;
            std::chrono::system_clock::time_point _last_report;
            typename stat<T>::function _function;
            bool _epoch_timing;
            unsigned int _last_epoch_number;
            unsigned int _count_all_epochs;
            T _last_error;
            std::thread _thread;
            bool _should_end;
        };
    }
}

#endif