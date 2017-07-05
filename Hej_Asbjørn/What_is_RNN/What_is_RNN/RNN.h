#pragma once

//////////////
//  RNN header
//////////////

/// INCLUDES ///

/// MACROS ///
#define _USE_MATH_DEFINES 


class RNN
{
public:
	RNN();
	//const int bptt_truncate;
	//const int inputDim;
	//const int outputDim;
	//const int hiddenDim;

	vector<double> Softmax(double * x, int xdim);
	double EncOHToOnehot(double * x,int dim); //IO uncertain
	double SGTrain(double * x, double * y, double learningRate); //IO uncertain

private:
	vector<double> FeedForward(double * x); //IO uncertain
	vector<double> FeedForwardForGradient(double * x, double time); //IO uncertain
	double dU(double * x, double * y, double timestep, double bptttrunc); //IO uncertain
	double dV(double * x, double * y, double timestep, double bptttrunc); //IO uncertain
	double dW(double * x, double * y, double timestep, double bptttrunc); //IO uncertain
	double dSdU(double * x, double timestep, double bptttrunc); //IO uncertain
	double dSdW(double * x, double timestep, double bptttrunc); //IO uncertain
	double FeedForwardForBatchGradient(double * x, double time); //IO uncertain

protected:

};