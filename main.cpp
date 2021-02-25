#include<iostream>
#include"LinearLayer.h"
#include"utility.h"
#include<vector>
#include<exception>

using MLVec = std::vector<double>;

int main()
{

	try
	{
		ML::LinearLayer fc(3, 3); //lin layer that takes 3 inputs and outputs 3 values
		std::cout << fc << '\n';

		auto out{ fc.forward({1.0,2.0,3.0}) };
		std::cout << "Output:\n";
		for (const auto& el : out)
			std::cout << el << ", ";

	}
	catch (const std::exception& exc)
	{
		std::cout << exc.what();
	}
	return 0;
}