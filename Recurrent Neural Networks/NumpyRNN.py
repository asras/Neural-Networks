import numpy as np
import time
import datetime
from tools import *

def Softmax(x):
	return np.exp(x)/np.sum(np.exp(x))

def EncOHToOnehot(x, dim):
	A = np.zeros(dim)
	A[x] = 1
	return np.array(A)


class NumpyRNN():

	def __init__(self, inputDim, outputDim, hiddenDim, bptt_truncate=4):
		self.U = np.random.uniform(-1/np.sqrt(inputDim), 1/np.sqrt(inputDim), (hiddenDim, inputDim))
		self.V = np.random.uniform(-1/np.sqrt(hiddenDim), 1/np.sqrt(hiddenDim), (outputDim, hiddenDim))
		self.W = np.random.uniform(-1/np.sqrt(hiddenDim), 1/np.sqrt(hiddenDim), (hiddenDim, hiddenDim))
		self.bptt_truncate = bptt_truncate
		self.inputDim = inputDim
		self.outputDim = outputDim
		self.hiddenDim = hiddenDim


	def FeedForward(self, x):
		U, W, V = self.U, self.W, self.V
		#x is encoded one-hot, i.e. [0, 19, 222, 1290, 1]
		o = []
		s = [np.zeros(self.hiddenDim)]
		for j in np.arange(len(x)):
			act = U[:,x[j]] + W.dot(s[j])
			st = np.tanh(act)
			s.append(st)
			o.append(Softmax(V.dot(st)))
		return np.array(o)

	def FeedForwardForGradient(self, x, time):
		U, W, V = self.U, self.W, self.V
		#x is encoded one-hot, e.g. [0, 19, 222, 1290, 1]
		o = []
		s = [np.zeros(self.hiddenDim)]
		acts = []
		for j in np.arange(time+1):			
			act = U[:,x[j]] + W.dot(s[j])
			st = np.tanh(act)
			acts.append(act)
			s.append(st)
			o.append(Softmax(V.dot(st)))
		return [np.array(o[-1]), np.array(s[-1]), np.array(acts[-1])]


	def dSdU(self, x, timestep,  bptttrunc):
		if timestep == 0 or bptttrunc==0:
			[o0, s0, act0] = self.FeedForwardForGradient(x, 0)
			x0 = EncOHToOnehot(x[0], self.inputDim)
			nameNotFound = np.outer(np.ones(len(act0))-(np.tanh(act0)**2), x0)
			return nameNotFound
		[ot, st, actt] = self.FeedForwardForGradient(x, timestep)
		xt = EncOHToOnehot(x[timestep], self.inputDim)
		dunnoName = self.W.dot(self.dSdU(x, timestep-1,  bptttrunc-1))
		nameDunno = np.ones(len(actt))-(np.tanh(actt)**2)
		dSdU = np.outer(nameDunno, xt)+np.diag(nameDunno).dot(dunnoName)
		return dSdU

	def dU(self, x, y, timestep, bptttrunc):
		yt = EncOHToOnehot(y[timestep], self.inputDim)
		[ot, st, actt] = self.FeedForwardForGradient(x, timestep)
		name404 = (ot-yt).dot(self.V).dot(self.dSdU(x, timestep,  bptttrunc))
		return name404


	def dV(self, x, y, timestep, bptttrunc):
		[ot, st, actt] = self.FeedForwardForGradient(x, timestep)
		yt = EncOHToOnehot(y[timestep], self.inputDim)
		return np.outer(ot-yt, st)

	def dSdW(self, x, timestep,  bptttrunc):
		if timestep == 0 or bptttrunc == 0:
			return np.zeros((self.hiddenDim, self.hiddenDim))

		[ot, st, actt] = self.FeedForwardForGradient(x, timestep)
		[opt, spt, actpt] = self.FeedForwardForGradient(x, timestep-1)
		diffActs = np.ones(len(actt))-np.tanh(actt)**2
		aGreatName = self.W.dot(self.dSdW(x, timestep-1,  bptttrunc-1))
		return np.outer(diffActs, spt) + np.diag(diffActs).dot(aGreatName)

	def dW(self, x, y, timestep, bptttrunc):
		[ot, st, actt] = self.FeedForwardForGradient(x, timestep)
		yt = EncOHToOnehot(y[timestep], self.inputDim)
		firstThing = (ot-yt).dot(self.V).dot(self.dSdW(x, timestep,  bptttrunc))
		return firstThing


	def SGTrain(self, x, y, learningRate):
		dU = np.zeros((self.hiddenDim, self.inputDim))
		dW = np.zeros((self.hiddenDim, self.hiddenDim))
		dV = np.zeros((self.outputDim, self.hiddenDim))
		for t in np.arange(len(x)):
			dU += self.dU(x, y, t, self.bptt_truncate)
			dW += self.dW(x, y, t, self.bptt_truncate)
			dV += self.dV(x, y, t, self.bptt_truncate)

		self.U += -learningRate*dU
		self.W += -learningRate*dW
		self.V += -learningRate*dV


	##Batch training - I'll keep the above for testing and reference even though
	##it will never be used on account of being so slow
	def FeedForwardForBatchGradient(self, X, time):
		U, W, V = self.U, self.W, self.V
		#X is matrix of encoded one-hot, e.g. [[0, 19, 222, 1290, 1], [0, 19, 222, 13, 1]]
		X = [[EncOHToOnehot(w) for w in s] for s in X]
		X = np.array(X).T
		o = []
		s = [np.array([np.zeros(self.hiddenDim)]*len(X)).T]
		acts = []
		for j in np.arange(time+1):			
			act = U.dot(X[:, j]) + W.dot(s[j])
			st = np.tanh(act)
			acts.append(act)
			s.append(st)
			o.append(Softmax(V.dot(st)))
		#This function returns matrices where each column corresponds to
		#output/hiddenstate/activation at timestep t
		return [np.array(o[-1]), np.array(s[-1]), np.array(acts[-1])]






if __name__=='__main__':
	t = datetime.datetime.now().time()
	print('Program started at', t)
	[word_to_index, index_to_word, X_train, Y_train] = load_training_data('NumpyRNNData.csv', 'NumpyRNNwordtofromindex.npz')
	word_dim = len(word_to_index)
	NN = NumpyRNN(word_dim, word_dim, 100, 4)
	Numpy_load_model_parameters('NumpySavedParameters.npz', NN)
	print('Training commenced.')
	t1 = time.time()
	for j in np.arange(1):
		NN.SGTrain(X_train[j], Y_train[j], 0.001)
	t2 = time.time()
	print('Training took ' + str(t2-t1) + ' seconds.')

	Numpy_save_model_parameters('NumpySavedParameters', NN)



	timeend = datetime.datetime.now().time()
	print('Program terminated at', timeend)



