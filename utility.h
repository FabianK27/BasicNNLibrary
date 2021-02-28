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


		template<typename T>
		double MSELoss_SingleEvent(const T& prediction, const T& ground_truth) // T must be iterable, usually will be vector<double>
		{
			// mse formula: 1/2n * sumx(||y(x) - a||^2), n batchsize; x: training samples
			assert(prediction.size() == ground_truth.size() && "size mismatch: prediction and ground truth of different dimesnion!");
			double loss{ 0.0 };
			for (std::size_t i{ 0 }; i < prediction.size(); ++i)
			{
				loss += std::pow(prediction.at(i) - ground_truth.at(i), 2);
			}
			return loss;
		}

	}
}