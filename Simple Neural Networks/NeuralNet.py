## Data for an n-layer neural network:
## Input dimension
## Output dimension
## n-1 numbers specifying the number of hidden neurons
## n matrices of appropriate dimensions
## an activation function and its derivate
## an output function - should be the canonical link so that the formula we use to calculate the gradient is valid



## We should use numpy as much as possible for efficient code



import numpy as np
import copy
import matplotlib.pyplot as plt
import math

def InstantiateRandomNbyMMatrix(N,M, RangeOfMagnitudes):
	randomMatrix = np.matrix(np.random.random((N,M)))
	randomMatrix = (randomMatrix-0.5)*2*RangeOfMagnitudes
	return randomMatrix


##There's no need for a neuron class because we dont want 
##to use different activation functions for different 
##neurons - otherwise we would make an implementation of the
##NN that has lists of neurons.
##class Neuron(object):
##
##	def __init__(self):
##		self.K = 3

def ActFunc1D(x):
	return np.tanh(x)

def DerivateActFunc1D(x):
	return 1-(np.tanh(x))**2

def ActivationFunction(x):
	if (len(x) == 1):
		return ActFunc1D(x)
	a = x
	for j in range(0,len(x)):
		a[j] = ActFunc1D(x[j])
	return a

def DerivateOfActivationFunction(x):
	if (len(x) == 1):
		return DerivateActFunc1D(x)
	a = x
	for j in range(0, len(x)):
		a[j] = DerivateActFunc1D(x[j])
	return a

def CanonicalLink(x):
	##print(type(x))
	##a = np.dot(x,x)
	return x

def ErrorFunction(actual, target):
	return 1/2*np.linalg.norm(actual-target)**2


def AppendBias(x):
	return np.insert(x, len(x), 1, axis=0)

def AppendZero(x):
	return np.insert(x, len(x), 0, axis=0)

def RemoveBiasDim(A):
	return A[:,0:-1]


##A Neural Network implementing on-line training
class NeuralNetwork(object):

	def __init__(self, _listOfDims):
		self.PastError = 100000000
		self.ListOfDims = _listOfDims 
		##A list of n+1 integers for an n-layer NN
		##The first entry is the input dimension
		##The last entry is the output dimension

		self.Layers = [] ##A list of matrices
		self.Gradients = [] ##A list of matrices
		self.TrainingRate = 0.01

		##Set up layers
		for j in range(0, len(self.ListOfDims)-1):
			inputDim = self.ListOfDims[j]
			outputDim = self.ListOfDims[j+1]
			self.Layers.append(InstantiateRandomNbyMMatrix(outputDim, inputDim+1,
				RangeOfMagnitudes=1))
			self.Gradients.append(np.matrix(np.zeros((outputDim, inputDim+1))))



	def Train(self, trainingData):
		index = np.random.randint(len(trainingData))
		return self.Train(trainingData, index)

	##Training data should be a list of tuples of vectors
	def Train(self, trainingData, indexToTrainOn):
		##Calculate all gradients		
		trainInput = trainingData[indexToTrainOn][0]
		trainOutput = trainingData[indexToTrainOn][1]
		actualOutput = self.FeedForward(trainInput)
		#if (self.ErrorFunction(trainingData)> self.PastError):
		#	print('-----Error increased!-----')
		#self.PastError = self.ErrorFunction(trainingData)
		if(indexToTrainOn % 200 == 0):
			print('Error is ', self.ErrorFunction(trainingData))
		for k in range(0, len(self.Layers)):
			gradx = self.CalculateGradXAtLayer(k, trainInput, trainOutput, actualOutput)
			gradMat = np.outer(gradx, self.FeedForwardToK(trainInput, K=k-1))
			self.Gradients[k] = np.matrix.copy(gradMat)


		##Update weights
		#self.TrainingRate = min(self.TrainingRate*self.ErrorFunction(trainingData)/7000, 0.002)
		for k in range(0, len(self.Layers)):
			self.Layers[k] += -self.TrainingRate*np.matrix.copy(self.Gradients[k])
		return

	def FeedForward(self, x):
		a = AppendBias(x)
		for k in range(0,len(self.Layers)-1):
			a = np.dot(self.Layers[k], a)
			a = ActivationFunction(a)
			a = AppendBias(a)
		a = np.dot(self.Layers[-1], a)
		a = CanonicalLink(a)
		return a

	def FeedForwardToK(self, x, K):
		if (K==-1):
			return AppendBias(x)
		a = AppendBias(x)
		for k in range(0,K+1):
			a = np.dot(self.Layers[k], a)
			a = ActivationFunction(a)
			a = AppendBias(a)
		return a

	def FeedForwardToKNoFinalActivation(self, x, K):
		if (K==-1):
			return AppendBias(x)
		a = AppendBias(x)
		for k in range(0,K):
			a = np.dot(self.Layers[k], a)
			a = ActivationFunction(a)
			a = AppendBias(a)
		a = np.dot(self.Layers[K], a)
		a = AppendBias(a)
		return a

	def CalculateGradXAtLayer(self, layerNumber,  trainInput, targetOutput, actualOutput):
		if (layerNumber == len(self.Layers)-1):
			##We use the canonical link by assumption:
			return actualOutput-targetOutput

		a = np.dot(self.CalculateGradXAtLayer(layerNumber+1, trainInput, targetOutput, 
			actualOutput).T,
			RemoveBiasDim(self.Layers[layerNumber+1])).T
		##We transpose a, take the matrix product with dz/dx(layer layerNumber)
		b = np.dot(a,self.DiffOfActivationFunctionAtLayer(trainInput, layerNumber).T)
		c = np.matrix(np.diag(b)).T ##This is necessary only because I don't know
		##how to calculate a vector d from vectors b,c w/ d_i = b_ic_i in any other
		##way (using numpy)
		if (c.shape[1] != 1):
			print('Dim errorDim errorDim errorDim error')

		if (b.shape[0] !=b.shape[1]):
			print('b is not square D:')
		return c

	def PrintGradients(self):
		print('-'*100)
		print(self.Gradients)
		print('-'*100)
		return

	def DiffOfActivationFunctionAtLayer(self, x, k):
		inputX = self.FeedForwardToKNoFinalActivation(x,k)
		outputX = np.matrix((np.zeros(self.Layers[k].shape[0]),1))
		outputX = DerivateOfActivationFunction(inputX)[0:-1]
		#outputX[-1] = 1

		return outputX

	def ErrorFunction(self, data):
		accError = 0
		for j in range(0, len(data)):
			y = self.FeedForward(data[j][0])
			t = data[j][1]
			accError += ErrorFunction(y,t)
		return accError



##Setup training data
NumberOfTrainingPts = 20
DataRange = 10
TrainingData = []
ErrorMagnitude = 0.1
maxX = -10*DataRange
minX = 10*DataRange
def trainfunc(x):
	return 0.1*x**2
for j in range(0, NumberOfTrainingPts):
	#x = np.matrix(np.random.random()*(DataRange-1))
	x = np.matrix([[j/NumberOfTrainingPts*DataRange]])
	if (x < minX):
		minX = x
	if (x > maxX):
		maxX = x
	y = np.matrix((trainfunc(x)+(np.random.random()-0.5)*2*ErrorMagnitude))
	TrainingData.append([x,y])

for j in range(0, len(TrainingData)):
	plt.plot(TrainingData[j][0], TrainingData[j][1], 'k.')

plt.plot([trainfunc(j) for j in range(math.floor(minX), math.ceil(maxX)+1)], 'k')


NeuralNet = NeuralNetwork(_listOfDims=[1,100, 20, 1])
NeuralNet.TrainingRate = 0.01/10
#NeuralNet.PrintGradients()


print('TRAINING STARTED')

NumberOfTrainingCycles = 2000
for j in range(0, NumberOfTrainingCycles):
	#print(notusedindex)
	#print(notusedindex % 200 == 0)
	NeuralNet.Train(TrainingData, j % NumberOfTrainingPts)

	#if ((j % 200) == 0):
		#NeuralNet.PrintGradients()

print('Final error is ', NeuralNet.ErrorFunction(TrainingData))
print('-'*50)
print(NeuralNet.Layers)
#print(NeuralNet.FeedForward(3))
#print([np.asscalar(NeuralNet.FeedForward(j)) for j in range(0, DataRange)])
plt.plot([np.asscalar(NeuralNet.FeedForward(np.matrix([[j]]))) for j in range(0, DataRange)], 'r')



#print('-'*50)
#print(NeuralNet.Layers)
#print('-'*50)

plt.show()




##TODO Replicate error where output of NN is order of magnitude 10^107 or whatever.
##TODO Why does NN sometimes return nan?





##TODO Implement batch training
##TODO Why does training work well for trainingdata drawn from tanh(x) and x
##but very poorly for x**2? 
##TODO Try some other functions









