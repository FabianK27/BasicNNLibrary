#pragma once
#include"neuron.h"
#include"constants.h"
#include<cassert>
#include<string>
#include<iterator>
#include<iostream>
namespace ML
{
	class LinearLayer
	{
	private:
		int m_numIn;
		int m_numOut;
		std::vector<Neuron> m_neurons;
	public:
		LinearLayer(int in, int out);
		
		std::vector<double> forward(const std::vector<double>& x) const;

		const std::vector<Neuron>& getNeurons() { return m_neurons; }

		friend std::ostream& operator<<(std::ostream& out, const LinearLayer& lin);
	};

}