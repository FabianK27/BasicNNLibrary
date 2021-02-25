#include "utility.h"

double ML::util::sigmoid(double z)
{
    return 1 / (1 + exp(-z));
}

double ML::util::getRandomdouble(double min, double max)
{
    auto dist = std::uniform_real_distribution<double>(min, max);
    return dist(ML::util::mt);
}
