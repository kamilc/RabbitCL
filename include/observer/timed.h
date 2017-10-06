#ifndef Timed_h
#define Timed_h

#include <memory>
#include <chrono>
#include <thread>
#include <deque>
#include <vector>
#include <boost/optional.hpp>
#include <functional>
#include <unordered_map>
#include <cstdlib>
#include <string>
#include "utilities.h"
#include "activation.h"
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
            timed& early_stop_when(std::function<bool(timed_observer<T>&)>);

            std::unique_ptr<base<T>> construct();
        private:
            std::chrono::duration<int> _interval;
            typename stat<T>::function _function;
            bool _epoch_timing;
            boost::optional<std::function<bool(timed_observer<T>&)>> _early_stop_when;
        };

        template<typename T>
        class timed_observer : public base<T>
        {
        public:
            timed_observer(timed<T>& config);
            void push_error(cost<T>&);
            void start_epoch(unsigned int, unsigned int);
            void start_batch(unsigned int, unsigned int);
            bool end_epoch(sequence<T>&);
            void push_outputs(activation<T>&, matrix<T>&);
            void start();
            void end();
            T last_error();
            T last_stat();
            T _last_stat_value;
        private:
            void main();
            std::string stat_name();
            std::chrono::duration<int> _interval;
            std::chrono::duration<double> _last_epoch_timing;
            std::chrono::system_clock::time_point _last_epoch_start;
            std::chrono::system_clock::time_point _last_report;
            std::deque<std::chrono::duration<double>> _epoch_timings;
            std::unordered_map<unsigned int, std::vector<T>> _errors;
            typename stat<T>::function _function;
            std::vector<stat<T>> _stats;
            boost::optional<std::function<bool(timed_observer<T>&)>> _early_stop_when;
            bool _epoch_timing;
            unsigned int _last_epoch_number;
            unsigned int _last_batch_number;
            unsigned int _count_all_epochs;
            unsigned int _count_all_batches;
            std::thread _thread;
            bool _should_end;
        };
    }
}

#endif
