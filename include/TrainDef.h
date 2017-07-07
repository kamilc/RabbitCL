#ifndef TrainDef_h
#define TrainDef_h

#include <valarray>
#include <boost/optional.hpp>
#include <boost/numeric/ublas/matrix.hpp>

class TrainDef {
private:
    boost::optional<boost::numeric::ublas::matrix<float>> _input;
    int _epochs = 100;
public:
    TrainDef(boost::numeric::ublas::matrix<float> input);

    boost::numeric::ublas::matrix<float> getInput();

    TrainDef& setEpochs(int epochs);
    int getEpochs();
};

#endif