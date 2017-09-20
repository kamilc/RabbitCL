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
        void timed_reporter<T>::main()
        {
            while(!this->_should_end)
            {
                auto now = std::chrono::system_clock::now();

                if(this->_last_epoch_number > 0 &&
                   std::chrono::duration<double, std::milli>(now - this->_last_report) > this->_interval)
                {
                    std::cout << "Epoch " << this->_last_epoch_number << "/" << this->_count_all_epochs;

                    std::cout << " " << this->_last_error;

                    if(this->_epoch_timing)
                    {
                        std::cout << " [ took: " << this->_last_epoch_timing.count() * 1000 << "ms ]";

                        auto epochs_left = this->_count_all_epochs - this->_last_epoch_number;
                        std::cout << " [ approx ETA: " << this->_last_epoch_timing.count() * epochs_left / 60 << " minutes ]";
                    }

                    std::cout << std::endl;

                    this->_last_report = std::chrono::system_clock::now();
                }
            }
        }

        template<typename T>
        void timed_reporter<T>::start()
        {
            std::cout << "Beginning optimization reported by the timed reporter (showing approximations)" << std::endl;

            this->_thread = std::thread(&timed_reporter<T>::main, this);
        }

        template<typename T>
        void timed_reporter<T>::end()
        {
            this->_should_end = true;
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
            this->_last_epoch_number = epoch;
            this->_count_all_epochs = count_all;
            this->_last_epoch_start = std::chrono::system_clock::now();
        }

        template<typename T>
        void timed_reporter<T>::end_epoch(sequence<T>& network)
        {
            if(this->_epoch_timing)
            {
                auto now = std::chrono::system_clock::now();
                this->_last_epoch_timing = std::chrono::duration<double, std::milli>(now - this->_last_epoch_start);
            }
        }

        INSTANTIATE(timed);
    }
}
