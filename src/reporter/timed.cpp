#include "reporter/timed.h"

namespace mozart
{
    namespace reporter
    {
        template<typename T>
        timed<T>::timed(std::chrono::duration<int> duration)
        {
            this->_interval = duration;
        }

        template<typename T>
        timed<T>& timed<T>::stats(typename stat<T>::function fn)
        {
            this->_function = fn;

            return *this;
        }

        template<typename T>
        timed<T>& timed<T>::epoch_timing(bool timing)
        {
            this->_epoch_timing = timing;

            return *this;
        }

        template<typename T>
        std::unique_ptr<base<T>> timed<T>::construct()
        {
            return std::unique_ptr<base<T>>(new timed_reporter<T>(*this));
        }

        template<typename T>
        timed_reporter<T>::timed_reporter(timed<T>& config)
        {
            this->_interval = config._interval;
            this->_function = config._function;
            this->_epoch_timing = config._epoch_timing;
        }

        template<typename T>
        void timed_reporter<T>::push_error(cost<T>& error)
        {
            this->_last_error = error.avg();
        }

        template<typename T>
        void timed_reporter<T>::start_epoch(unsigned int epoch, unsigned int count_all)
        {
            this->_last_epoch_start = std::chrono::system_clock::now();
        }

        INSTANTIATE(timed);
    }
}
