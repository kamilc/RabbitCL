#include "observer/timed.h"
#include "rang.hpp"

namespace mozart
{
    namespace observer
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
            return std::unique_ptr<base<T>>(new timed_observer<T>(*this));
        }

        template<typename T>
        void timed_observer<T>::main()
        {
            while(!this->_should_end)
            {
                auto now = std::chrono::system_clock::now();

                if(this->_last_epoch_number > 0 &&
                   std::chrono::duration<double, std::milli>(now - this->_last_report) > this->_interval)
                {
                    std::cout << rang::fg::gray << "Epoch:" << rang::fg::reset << std::right << std::setw(6) << this->_last_epoch_number << "/" << this->_count_all_epochs;

                    std::cout << " | " << rang::fg::gray << "Error:" << rang::fg::reset << std::right << std::setw(14) << this->_last_error;

                    if(this->_epoch_timing)
                    {
                        std::cout << " | " << rang::fg::gray << "Epoch timing:" << rang::fg::reset << std::right << std::setw(6) << (int)(this->_last_epoch_timing.count() * 1000) << "ms";

                        auto epochs_left = this->_count_all_epochs - this->_last_epoch_number;
                        auto epochs_size = this->_epoch_timings.size();
                        double sum_epochs = 0;

                        for(auto i = 0; i < epochs_size; i++)
                        {
                            sum_epochs += this->_epoch_timings[i].count();
                        }

                        auto left = (sum_epochs / epochs_size) * epochs_left / 60;
                        int minutes = (int)left;
                        int seconds = 60 * (left - minutes);

                        std::cout << " | " << rang::fg::gray << "ETA:" << rang::fg::reset << std::setw(5) << minutes << ":" << std::setw(2) << std::setfill('0') << std::right << seconds << std::setfill(' ');
                    }

                    std::cout << std::endl;

                    this->_last_report = std::chrono::system_clock::now();
                }
            }
        }

        template<typename T>
        void timed_observer<T>::start()
        {
            std::cout << "Beginning optimization reported by the timed observer (showing approximations)" << std::endl << std::endl;

            this->_thread = std::thread(&timed_observer<T>::main, this);
        }

        template<typename T>
        void timed_observer<T>::end()
        {
            this->_should_end = true;
        }

        template<typename T>
        timed_observer<T>::timed_observer(timed<T>& config)
        {
            this->_interval = config._interval;
            this->_function = config._function;
            this->_epoch_timing = config._epoch_timing;
        }

        template<typename T>
        void timed_observer<T>::push_error(cost<T>& error)
        {
            this->_last_error = error.avg();
        }

        template<typename T>
        void timed_observer<T>::start_epoch(unsigned int epoch, unsigned int count_all)
        {
            this->_last_epoch_number = epoch;
            this->_count_all_epochs = count_all;
            this->_last_epoch_start = std::chrono::system_clock::now();
        }

        template<typename T>
        void timed_observer<T>::end_epoch(sequence<T>& network)
        {
            if(this->_epoch_timing)
            {
                auto now = std::chrono::system_clock::now();
                this->_last_epoch_timing = std::chrono::duration<double, std::milli>(now - this->_last_epoch_start);
                this->_epoch_timings.push_back(this->_last_epoch_timing);
                if(this->_epoch_timings.size() > 100)
                {
                    this->_epoch_timings.pop_front();
                }
            }
        }

        INSTANTIATE(timed);
    }
}
