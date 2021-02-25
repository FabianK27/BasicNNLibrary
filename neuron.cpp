#include "neuron.h"
namespace ML
{
    double Neuron::forward(const std::vector<double>& x) const
    {
        return util::dot<std::vector<double>>(m_weights, x) + m_bias;
    }

    void Neuron::uniformRandomInit(double min, double max)
    {
 
        for (int i{ 0 }; i < m_inputs; ++i)
        {
            m_weights.push_back(util::getRandomdouble(min, max));
        }
        
        m_bias = util::getRandomdouble(min, max);
        
    }

    std::ostream& operator<<(std::ostream& out, const ML::Neuron& neuron)
    {
        out << "Weights: ";
        for (const auto& weight : neuron.m_weights)
        {
            out << weight << ", ";
        }
        out << "Bias: " << neuron.m_bias;

        return out;
    }

}