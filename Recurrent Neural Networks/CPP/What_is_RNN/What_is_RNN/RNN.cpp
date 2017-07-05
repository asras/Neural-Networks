//////////////
// Cpp implementation of RNN
//////////////

/// INCLUDES ///
#include<iostream>
#include<string>
#include<cmath>
#include "RNN.h"

/// NAMESPACES ///
using namespace std;


RNN::RNN() //overriden default constructor. Similar to initializer in python
{
	//TODO implementation
}

vector<double> RNN::Softmax(double * x, int xdim)
{
	vector<double> X;
	double sum = 0;

	for (int i = 0; i < xdim; i++)
		sum += exp(x[i]);

	for (int i = 0; i < 5; i++)
		X[i] = exp(x[i]) / sum;

	return X;
}

double RNN::EncOHToOnehot(double x[], int dim)
{
	//TODO: do implemetation
	return 0.0;
}

