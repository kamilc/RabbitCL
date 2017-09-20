#include "reporter/timed.h"

namespace mozart
{
    namespace reporter
    {
        template<typename T>
        timed<T>::timed(double ms)
        {
            this->_interval = ms;
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

        INSTANTIATE(timed);
    }
}
