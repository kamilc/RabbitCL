#include "stats/accuracy.h"

namespace mozart
{
    namespace stats
    {
        template<typename T>
        stat<T> accuracy(matrix<T>& outputs, matrix<T>& targets)
        {
            matrix<T> predicted = squashmax<T>(outputs);

            stat<T> accuracy;

            accuracy.name = "Accuracy";
            accuracy.count = predicted.size1();
            accuracy.out = accuracy_rate<T>(predicted, targets);

            return accuracy;
        }

        template stat<float> accuracy(matrix<float>& predicted, matrix<float>& targets);
        template stat<double> accuracy(matrix<double>& predicted, matrix<double>& targets);
    }
}
