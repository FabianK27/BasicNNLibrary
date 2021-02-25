#include<iostream>
#include"LinearLayer.h"
#include"utility.h"
#include<vector>
#include<exception>
#include"Model.h"

using MLVec = std::vector<double>;

class MyBasicModel : public ML::Model
{
public:
	ML::LinearLayer fc1{ 3, 10 };
	ML::LinearLayer fc2{ 10, 3 };

	std::vector<double> forward(std::vector<double> x) override
	{
		x = fc1(x);
		x = ML::util::sigmoid(x);
		x = fc2(x);
		x = ML::util::sigmoid(x);
		return x;
	}

};

int main()
{


	try
	{
		MyBasicModel model;
		auto out = model.forward({ 1.0,2.0,3.0 });
		for (const auto& element : out)
			std::cout << "Out tensor element: " << element << ", ";
	}
	catch (const std::exception& exc)
	{
		std::cout << exc.what();
	}
	return 0;
}