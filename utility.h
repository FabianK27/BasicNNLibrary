#pragma once
#include<cmath>
#include<cassert>
#include<random>
#include<ctime>
namespace ML
{
	namespace util
	{
		double sigmoid(double x);
		std::vector<double> sigmoid(std::vector<double>& x);

		template<typename T>
		double dot(const T& t1, const T& t2)
		{
			// return the dot product of the inputs
			assert(t1.size() == t2.size() && "Can only take dot product between equal-dimension vectors");
			double dotSum = 0.0;
			for (std::size_t i{ 0 }; i < t1.size(); ++i)
			{
				dotSum += t1[i] * t2[i];
			}
			return dotSum;
		}

		// init a random machine
		static std::mt19937 mt{ static_cast<std::mt19937::result_type>(time(nullptr)) };

		double getRandomdouble(double min, double max);


	}
}