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

            size_t count_correct = 0;

            for(auto row = 0; row < predicted.size1(); row++)
            {
                bool correct = true;

                for(auto column = 0; column < predicted.size2(); column++)
                {
                    if(predicted(row, column) != targets(row, column))
                    {
                        correct = false;
                        break;
                    }
                }

                if(correct)
                {
                    count_correct += 1;
                }
            }

            accuracy.out = (T)count_correct/(T)predicted.size1();

            return accuracy;
        }

        template stat<float> accuracy(matrix<float>& predicted, matrix<float>& targets);
        template stat<double> accuracy(matrix<double>& predicted, matrix<double>& targets);
    }
}
