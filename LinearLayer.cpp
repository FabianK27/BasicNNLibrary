#include "LinearLayer.h"

namespace ML
{
	LinearLayer::LinearLayer(int in, int out) : m_numIn{ in }, m_numOut{ out }
	{
		m_neurons.reserve(m_numOut);
		for (int i{ 0 }; i < m_numOut; ++i)
			m_neurons.push_back(Neuron(m_numIn));
	}

	std::vector<double> LinearLayer::forward(const std::vector<double>& x) const
	{
		assert(x.size() == m_numIn && "Dimension Mismatch!");
		std::vector<double> out;
		out.reserve(m_numOut);
		for (int i{ 0 }; i < m_numOut; ++i)
		{
			out.push_back(m_neurons.at(i).forward(x));
		}
		return out;
	}

	std::ostream& operator<<(std::ostream& out, const LinearLayer& lin)
	{
		for (auto it{ lin.m_neurons.begin() }; it != lin.m_neurons.end(); ++it)
			out << *it << '\n';
		return out;
	}

}