#include "stdafx.h" //it wont compile without this, even though it's included in RNN.h...
#include "RNN.h"

RNN::RNN(int inputDim, int hiddenDim, int outputDim) {
	float a = 1. / (sqrt((float)hiddenDim));
	float b = 1. / (sqrt((float)outputDim)); //Am I overdoing it with the casts here?
	U = MatrixXf::Random(hiddenDim, inputDim)*a;
	W = MatrixXf::Random(hiddenDim, hiddenDim)*a;
	V = MatrixXf::Random(outputDim, hiddenDim)*b;
	trainData = TrainStruct(0);
}

RNN::~RNN() {
	trainData.~TrainStruct(); //Is this necessary?
}


void RNN::Train(VectorXf x, VectorXf t) {
	//Let L be the length of t = length of x
	//outputs = FeedForward(x).outputs;
	//hiddenStates = FeedForward(x).hiddenStates;
	//for k = 0 to L-1 calculate
	//delta = (y-outputs[k]).transpose -- NB! this may be the way you actually transpose
	//dV += delta.transpose outer product with hiddenStates[k]
	//for j = k; j >= 0; j--	
	//delta = delta*V*(1 + tanh(s[j])^2)(all elementwise)
	//dU += delta.transpose outer product with x(k).OneHotToVector
	//dW += delta.transpose outer product with hiddenStates[j]
	//delta = delta*W

	//Something like this, should double-check


	throw;
}

void RNN::Train(VectorXf * X, int numOfX, VectorXf * T, int numOfT) {
	throw;
}




TrainStruct RNN::FeedForward(VectorXf * X, int numOfInput) {
	//Like the other function but with U*X[i] instead of U.col(x(i))
	throw;
}


TrainStruct RNN::FeedForward(VectorXf x) {
	int lenOfSeq = x.rows() >= x.cols() ? x.rows(): x.cols();
	
	trainData = TrainStruct(lenOfSeq);

	for (int i = 0; i < lenOfSeq; i++) {
		VectorXf activation = U.col(x(i)) + W*trainData.hiddenStates[0];
		VectorXf hiddenState = tanh(activation.array());
		trainData.hiddenStates[i + 1] = hiddenState;
		trainData.outputs[i] = SoftMax(V*hiddenState);
	}	
	return trainData;
}

VectorXf RNN::SoftMax(VectorXf s) {
	return inverse(1 + exp(-s.array()));
}