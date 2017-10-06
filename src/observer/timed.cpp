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
                   std::chrono::duration<double, std::milli>(now - this->_last_report) > this->_interval &&
                   this->_last_batch_number + 1 == this->_count_all_batches)
                {
                    std::cout << rang::fg::gray << "Epoch:" << rang::fg::reset << std::right << std::setw(6) << this->_last_epoch_number << "/" << this->_count_all_epochs;

                    std::cout << " | " << rang::fg::gray << "Error:" << rang::fg::reset << std::right << std::setw(14) << this->last_error();

                    std::cout << " | " << rang::fg::gray << this->stat_name() << ":" << rang::fg::reset << std::right << std::setw(14) << this->_last_stat_value;

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
            if(!this->_epoch_timing) return;

            if(this->_errors.find(this->_last_epoch_number) == this->_errors.end())
            {
                if(this->_count_all_batches > 0)
                {
                    this->_errors[this->_last_epoch_number] = std::vector<T>(this->_count_all_batches);
                }
                else
                {
                    this->_errors[this->_last_epoch_number] = std::vector<T>();
                }
            }

            this->_errors[this->_last_epoch_number].push_back(error.avg());
        }

        template<typename T>
        T timed_observer<T>::last_error()
        {
            if(this->_errors.find(this->_last_epoch_number - 1) == this->_errors.end())
            {
                return 0;
            }

            T sum = 0;

            for(auto element : this->_errors[this->_last_epoch_number - 1])
            {
                sum += element;
            }

            return sum / this->_count_all_batches;
        }

        template<typename T>
        T timed_observer<T>::last_stat()
        {
            size_t count_all = 0;
            T result = 0;

            for(auto stat : this->_stats)
            {
                count_all += stat.count;
            }

            for(auto stat : this->_stats)
            {
                result += stat.out * ((T)stat.count/(T)count_all);
            }

            return result;
        }

        template<typename T>
        std::string timed_observer<T>::stat_name()
        {
            return this->_stats[0].name;
        }

        template<typename T>
        void timed_observer<T>::start_epoch(unsigned int epoch, unsigned int count_all)
        {
            this->_last_epoch_number = epoch;
            this->_count_all_epochs = count_all;
            this->_last_epoch_start = std::chrono::system_clock::now();
        }

        template<typename T>
        void timed_observer<T>::start_batch(unsigned int batch, unsigned int count_all)
        {
            this->_last_batch_number = batch;
            this->_count_all_batches = count_all;
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

            this->_last_stat_value = this->last_stat();
            this->_stats.clear();
        }

        template<typename T>
        void timed_observer<T>::push_outputs(activation<T>& outputs, matrix<T>& targets)
        {
            this->_stats.push_back(this->_function(outputs.out, targets));
        }

        INSTANTIATE(timed);
    }
}
