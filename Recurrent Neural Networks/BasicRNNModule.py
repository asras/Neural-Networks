import numpy as np
import theano
import theano.tensor as T


class HiddenLayer(object):

	def __init__(self, inputDim, outputDim, inputVar = None, bptt_truncate = 4):
		U = np.random.uniform(-np.sqrt(1./inputDim), np.sqrt(1./inputDim), (outputDim, inputDim))
		W = np.random.uniform(-np.sqrt(1./outputDim), np.sqrt(1./outputDim), (outputDim, outputDim))
		self.U = theano.shared(value=U.astype(theano.config.floatX))
		self.W = theano.shared(value=W.astype(theano.config.floatX))
		self.outputDim = outputDim
		self.inputDim = inputDim
		self.TrainingRate = 0.01
		self.bptt_truncate = bptt_truncate
		if (inputVar == None):
			##Network element is the first in the network
			self.InputVar = T.dmatrix()
		else:
			self.InputVar = inputVar
		
		self.Build()

	def Build(self):
		##Build the computation graph for this element of the network
		
		x = self.InputVar
		def ff(xt,st_prev):
			activation = self.U.dot(xt) + self.W.dot(st_prev)
			hiddenState = T.tanh(activation)
			return hiddenState
		
		s, updates = theano.scan(ff, sequences=x, 
			outputs_info = np.zeros(self.outputDim),
			 truncate_gradient = self.bptt_truncate)
		##When using a matrix as a sequence it uses the rows as input
		self.OutputVar = s
		##Only need the function if element is the last in network
		self.FeedForward = theano.function([x], s)

		
		def FFGraph(inputseq):
			t, y = theano.scan(ff, sequences=inputseq,
			 outputs_info = np.zeros(self.outputDim), 
			 truncate_gradient=self.bptt_truncate)
			return t
		##When scanning over a 3-tensor it again scans over rows,
		##e.g. w/ [[[1,1],[1,1]],[[2,2],[2,2]]] first we get 1s then 2s
		S = T.tensor3()
		q, updates = theano.scan(FFGraph, sequences=S)
		self.BatchOutputVar = q
		##Only need the function if element is the last in network
		self.BatchFeedForward = theano.function([S], q)

	def GetGradient(self, costfunction, xvar, tvar):
		##Run this after full network and cost-function is set up
		##to get graphs and functions needed for training
		self.dU = T.grad(costfunction, self.U)
		self.dW = T.grad(costfunction, self.W)
		##The actual function
		print('Trainingrate: ', self.TrainingRate)
		self.GDTrain = theano.function([xvar, tvar], [], 
			updates =[(self.U, self.U - self.TrainingRate*self.dU),
			(self.W, self.W - self.TrainingRate*self.dW)])
		print('Stored training function')
	def Train(self, input, target):
		self.GDTrain(input, target)
		






if __name__=='__main__':
	#####Test code#####

	##Test that instantiating works
	Atest = HiddenLayer(3,3)

	##Test that U and W work
	print(Atest.U.get_value())
	print(Atest.W.get_value())


	print('##'*30)


	##Test that gradients return zero when given trivial costfct
	x = T.scalar()
	d = T.dmatrix('d')
	m = T.dmatrix('m')
	cost = (T.sum(Atest.U.dot(d).dot(m))+T.sum(Atest.W))
	Atest.TrainingRate = 1000
	Atest.GetGradient(cost, d, m)
	a = np.zeros((3,3))
	b = np.zeros((3,3))

	
	print('W before training: ', Atest.W.get_value()) 
	Atest.GDTrain(a,b)
	print('W after training: ', Atest.W.get_value())
	#testfunc = theano.function([x], Atest.dU)
	#print(testfunc(0))
	#print(testfunc(1))	
	##Some sort of exception is thrown if fct in T.grad(fct, x)
	##does not depend on x (if x is disconnected from fct's graph)

	##Construct with input variable
	k = T.dmatrix()
	Atest2 = HiddenLayer(3,3, k)