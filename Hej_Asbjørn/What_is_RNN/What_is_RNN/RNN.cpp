//////////////
// Cpp implementation of RNN
//////////////

/// INCLUDES ///
#include<iostream>
#include<string>
#include<cmath>
#include<vector>
#include<array>
#include<algorithm>
#include <random>
#include<chrono>
#include "RNN.h"

/// NAMESPACES ///
using namespace std;


RNN::RNN() //overriden default constructor. Similar to initializer in python
{	
	const int bptt_truncate = 4;
	const int inputDim = 10;
	const int outputDim = 10;
	const int hiddenDim = 5;

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count(); //construct a time-based seed:
	std::default_random_engine generator(seed);	//construct random generator
	std::uniform_real_distribution<double> distributionU(-1/sqrt(inputDim), 1/sqrt(inputDim));
	std::uniform_real_distribution<double> distributionV(-1/sqrt(hiddenDim), 1/sqrt(hiddenDim));
	std::uniform_real_distribution<double> distributionW(-1/sqrt(hiddenDim), 1/sqrt(hiddenDim));
	double U[hiddenDim][inputDim];
	double V[outputDim][hiddenDim];
	double W[hiddenDim][hiddenDim];

	for (int i = 0; i < hiddenDim; i++) {
		for (int j = 0; j < inputDim; j++) {
			U[i][j] = distributionU(generator);
			V[i][j] = distributionV(generator);
		}	W[i][j] = distributionW(generator);
	}
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

double RNN::EncOHToOnehot(double * x, int dim)
{
	//TODO: do implemetation
	return 0.0;
}

double RNN::SGTrain(double * x, double * y, double learningRate)
{
	//TODO: do implemetation
	return 0.0;
}

vector<double> RNN::FeedForward(double * x)
{
	//TODO: do implemetation
	return vector<double>();
}

double RNN::FeedForwardForBatchGradient(double * x, double time)
{
	//TODO: do implemetation
	return 0.0;
}

vector<double> RNN::FeedForwardForGradient(double * x, double time)
{
	//TODO: do implemetation
	return vector<double>();
}

double RNN::dU(double * x, double * y, double timestep, double bptttrunc)
{
	//TODO: do implemetation
	return 0.0;
}

double RNN::dV(double * x, double * y, double timestep, double bptttrunc)
{
	//TODO: do implemetation
	return 0.0;
}

double RNN::dW(double * x, double * y, double timestep, double bptttrunc)
{
	//TODO: do implemetation
	return 0.0;
}

double RNN::dSdU(double * x, double timestep, double bptttrunc)
{
	return 0.0;
}

double RNN::dSdW(double * x, double timestep, double bptttrunc)
{
	//TODO: do implemetation
	return 0.0;
}



