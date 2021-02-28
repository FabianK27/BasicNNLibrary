#pragma once
#include<iostream>
namespace ML
{
	template<class T>
	class DataLoader
	{
		// A class to hold a given dataset and return (shuffled) mini batches
	private:
		T m_dataset;
		int m_len;
		int m_batchsize;
		bool m_shuffle;

	public:
		DataLoader(T data, int len, int batchsize, bool shuffle);
	};

	template<class T>
	DataLoader<T>::DataLoader(T data, int len, int batchsize, bool shuffle) : m_dataset{ data }, m_len{ len }, m_batchsize{ batchsize }, m_shuffle{ shuffle }
	{
		std::cout << "Loaded dataset with " << m_len << " elements.\n";
	}

//#include"DataLoader.cpp" //so that we can split declaration and definition even in template case


}