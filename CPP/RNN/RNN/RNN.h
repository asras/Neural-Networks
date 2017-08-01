#pragma once


#include "stdafx.h"
struct TrainStruct //What a shitty name
{
	VectorXf * hiddenStates; //We have to do VectorXf because the dimension varies
	VectorXf * outputs;


	TrainStruct() {};
	TrainStruct(int lenOfSeq) {
		hiddenStates = new VectorXf[lenOfSeq + 1];
		outputs = new VectorXf[lenOfSeq];
	}
	

	~TrainStruct() {
		delete[] hiddenStates; delete[] outputs;
	}
};




//We'll stick to floats for this one bro
class RNN 
{

public:
	RNN(int inputDim, int hiddenDim, int outputDim);
	~RNN();
	void Train(VectorXf * X, int numOfX, VectorXf * T, int numOfT); //X is input, T is target
	void Train(VectorXf x, VectorXf t); //Training on one-hot encoded vectors
	TrainStruct FeedForward(VectorXf* X, int numOfInput); //Input array of not-encoded vectors, return an array of outputs. PHILIP: Is it necessary to have 
	//the number of elements in the array as an input?
	TrainStruct FeedForward(VectorXf x); //input: one-hot encoded vector. Output trainstruct which contains arrays of calculated
	//hidden states and outputs
	 

private:
	MatrixXf U; //Dimensions: hiddenDim x inputDim
	MatrixXf W; //Dimensions: hiddenDim x hiddenDim
	MatrixXf V; //Dimensions: outputDim x hiddenDim
	TrainStruct trainData; //What a shitty name
	VectorXf SoftMax(VectorXf s);

};


