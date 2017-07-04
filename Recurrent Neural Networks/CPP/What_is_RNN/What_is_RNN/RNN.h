#pragma once

//////////////
//  RNN header
//////////////

/// INCLUDES ///
#include<string>
#include<cmath>
#include<array>
#include<vector>
#include<algorithm>

/// MACROS ///
#define _USE_MATH_DEFINES 

class RNN
{
public:
	RNN();
	double U, V, W; //is it double?
	double bptt_truncate; //is it double?
	double outputDim, hiddenDim; //is it double?

	vector<double> RNN::Softmax(double* x, int xdim);
	double EncOHToOnehot(double x[],int dim); //IO uncertain

private:
	vector<double> FeedForward(double x[]); //IO uncertain
	vector<double> FeedForwardForGradient(double x[], double time); //IO uncertain
	double dSdU(double x[], double timestep, double bptttrunc); //IO uncertain
	
	double dU(double* x, double* y, double timestep, double bptttrunc); //IO uncertain
	double def dV(double* x, double*  y, double timestep, double bptttrunc); //IO uncertain
	double def dSdW(double* x, double timestep, double bptttrunc); //IO uncertain
	double def dW(double* x, double* y, double timestep, double bptttrunc); //IO uncertain
	double def SGTrain(double* x, double* y, double learningRate); //IO uncertain
	double def FeedForwardForBatchGradient(double* X, double time); //IO uncertain

protected:

};