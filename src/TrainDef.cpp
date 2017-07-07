#include "TrainDef.h"

TrainDef::TrainDef(boost::numeric::ublas::matrix<float> input)
{
    this->_input = input;
}

TrainDef& TrainDef::setEpochs(int epochs)
{
    this->_epochs = epochs;

    return *this;
}

int TrainDef::getEpochs()
{
    return this->_epochs;
}

boost::numeric::ublas::matrix<float> TrainDef::getInput()
{
    return this->_input.get();
}