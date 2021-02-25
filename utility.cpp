#include "utility.h"

double ML::util::sigmoid(double z)
{
    return 1 / (1 + exp(-z));
}

std::vector<double> ML::util::sigmoid(std::vector<double>& x)
{
    std::vector<double> out;
    out.reserve(x.size());
    for (const auto& el : x)
    {
        out.push_back(sigmoid(el));
    }
    return out;
}

double ML::util::getRandomdouble(double min, double max)
{
    auto dist = std::uniform_real_distribution<double>(min, max);
    return dist(ML::util::mt);
}
