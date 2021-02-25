#ifndef NEURON_H
#define NEURON_H
#include<cassert>
#include<vector>
#include"utility.h"
#include"constants.h"
#include<iostream>


namespace ML
{
	// forward declare LinearLayer class
	class  LinearLayer;

	class Neuron
	{
	private:
		std::vector<double> m_weights;
		double m_bias{ 0.0 };
		int m_inputs;

		Neuron( int inputs, double bias = 0.0) :  m_bias{ bias }, m_inputs{inputs}
		{
			//resize vector to hold inputs number of weights
			m_weights.reserve(inputs);
			//and randomly initialize it
			uniformRandomInit(-RANDOMINIT_LIMIT, RANDOMINIT_LIMIT);
		}

		// setters and getters
		const std::vector<double>& getWeights() const { return m_weights; }
		void setWeigthtatIndex(int index, double w) { m_weights.at(index) = w; }
		void setWeights(const std::vector<double>& v) 
		{
			assert(v.size() == m_weights.size() && "Vector dimension mismatch!");
			m_weights = v; 
		}
		double getBias() const { return m_bias; }
		void setBias(double b) { m_bias = b; }
		// ML utility
		double forward(const std::vector<double>& x) const;

		void uniformRandomInit(double min, double max);

		friend std::ostream& operator<< (std::ostream& out, const ML::Neuron& neuron);
		//make LinearLayer a friend class so that it can instantiate neurons
		friend class LinearLayer;
	};

}
#endif
