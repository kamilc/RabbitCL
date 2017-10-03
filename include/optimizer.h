#ifndef Optimizer_h
#define Optimizer_h

#include "matrix.h"
#include "utilities.h"
#include "sequence.h"
#include "observer.h"

namespace mozart
{
    template<typename T>
    class optimizer
    {
    public:
        virtual void run(sequence<T> &network, matrix<T> &data, matrix<T> &targets) = 0;

        optimizer& epochs(unsigned long epochs)
        {
            this->_epochs = epochs;

            return *this;
        }

        optimizer& batches(unsigned long batches)
        {
            this->_batches = batches;

            return *this;
        }

        optimizer& push_observer(mozart::observer::config<T>& config)
        {
            this->_observers.push_back(std::move(config.construct()));

            return *this;
        }
    protected:
        unsigned long  _epochs;
        unsigned long  _batches;
        typename cost<T>::function _cost;
        vector<unique_ptr<mozart::observer::base<T>>> _observers;
    };
}

#endif
