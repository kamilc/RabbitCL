#include "Layer.h"
#include "IdentityLayer.h"

IdentityLayer::IdentityLayer() : Layer(0, IdentityActivation())
{

}
