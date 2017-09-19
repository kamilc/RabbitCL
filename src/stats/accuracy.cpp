#include "stats/accuracy.h"

namespace mozart
{
    namespace stats
    {
        template<typename T>
        stat<T> accuracy(matrix<T>& predicted, matrix<T>& targets)
        {
            // todo: implement me
            return stat<T>();
        }

        template stat<float> accuracy(matrix<float>& predicted, matrix<float>& targets);
        template stat<double> accuracy(matrix<double>& predicted, matrix<double>& targets);
    }
}
