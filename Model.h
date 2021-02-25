#pragma once
#include"LinearLayer.h"
#include"constants.h"
#include"utility.h"

namespace ML
{
	class Model
	{
	private:
		
	public:
		

		virtual std::vector<double> forward(std::vector<double> x) = 0;
	};

}