import numpy as np
import time
import datetime
from tools import *

def Softmax(x):
	return np.exp(x)/np.sum(np.exp(x))

def EncOHToOnehot(x, dim):
	A = np.zeros(dim)
	A[x] = 1
	return A


class NumpyRNN():

	def __init__(self, inputDim, outputDim, hiddenDim, bptt_truncate):
		U = np.random.uniform(-1/np.sqrt(inputDim), 1/np.sqrt(inputDim), (hiddenDim, inputDim))
		V = np.random.uniform(-1/np.sqrt(hiddenDim), 1/np.sqrt(hiddenDim), (outputDim, hiddenDim))
		W = np.random.uniform(-1/np.sqrt(hiddenDim), 1/np.sqrt(hiddenDim), (hiddenDim, hiddenDim))
		self.bptt_truncate = bptt_truncate
		self.inputDim = inputDim
		self.outputDim = outputDim
		self.hiddenDim = hiddenDim


	def FeedForward(self, x):
		U, W, V = self.U, self.W, self.V
		#x is encoded one-hot, i.e. [0, 19, 222, 1290, 1]
		o = []
		s = [np.zeros(np.zeros(self.hiddenDim))]
		for j in np.arange(len(x)):
			act = U[:,x[j]] + W.dot(s[j])
			st = np.tanh(act)
			s.append(st)
			o.append(Softmax(V.dot(st)))
		return o

	def FeedForwardForGradient(self, x, time):
		U, W, V = self.U, self.W, self.V
		#x is encoded one-hot, i.e. [0, 19, 222, 1290, 1]
		o = []
		s = [np.zeros(np.zeros(self.hiddenDim))]
		acts = []
		for j in np.arange(time+1):			
			act = U[:,x[j]] + W.dot(s[j])
			st = np.tanh(act)
			acts.append(act)
			s.append(st)
			o.append(Softmax(V.dot(st)))
		return [o[-1], s[-1], act[-1]]


	def dSdU(self, x, timestep):
		if timestep == 0:
			[o0, s0, act0] = self.FeedForwardForGradient(x, 0)
			x0 = EncOHToOnehot(x[0])
			nameNotFound = np.outer(np.ones(len(act0))-(np.tanh(act0)**2), x0)
			return nameNotFound
		[ot, st, actt] = self.FeedForwardForGradient(x[0:timestep], timestep)
		xt = EncOHToOnehot(x[timestep])
		dunnoName = self.W.dot(self.dSdU(x, y, timestep-1))
		nameDunno = np.ones(len(actt))-(np.tanh(actt))**2
		dSdU = np.outer(nameDunno, xt+dunnoName)
		return dSdU

	def dU(self, x, y, timestep):
		yt = EncOHToOnehot(y[timestep])
		ot = self.FeedForwardForGradient(x, timestep)
		name404 = (ot-yt).dot(self.V).dot(self.dSdU)
		return name404


	def dV(self, x, y, timestep):
		[ot, st, actt] = self.FeedForwardForGradient(x, timestep)
		yt = EncOHToOnehot(y[timestep])
		return np.outer(ot-yt, st)

	def dSdW(self, x, timestep):
		if timestep == 0:
			return np.zeros(self.hiddenDim, self.hiddenDim)

		[ot, st, actt] = self.FeedForwardForGradient(x, timestep)
		[opt, spt, actpt] = self.FeedForwardForGradient(x, timestep-1)
		diffActs = np.ones(len(actt))-np.tanh(actt)**2
		aGreatName = spt + self.W.dot(self.dSdW(x, timestep-1))
		return np.outer(diffActs, aGreatName)

	def dW(self, x, y, timestep):
		[ot, st, actt] = self.FeedForwardForGradient(x, timestep)
		yt = EncOHToOnehot(y[timestep])
		firstThing = (ot-yt).dot(self.V).self.dSdW
		return firstThing


	def SGTrain(self, x, y, learningRate):
		dU = np.zeros(self.hiddenDim, self.inputDim)
		dW = np.zeros(self.hiddenDim, self.hiddenDim)
		dV = np.zeros(self.outputDim, self.hiddenDim)
		for t in np.arange(len(x)):
			dU += self.dU(x, y, t)
			dW += self.dW(x, y, t)
			dV += self.dV(x, y, t)

		self.U += learningRate*dU
		self.W += learningRate*dW
		self.V += learningRate*dV
		




