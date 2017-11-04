import theano.tensor as T
import theano
import numpy as np
import csv
import time #Time is money
from tools import *
import sys
import datetime


class RNN(object):

	def __init__(self, inputDim, outputDim, hiddenDim = 100, bptt_trunc= 4):
		U = np.random.uniform(-1/np.sqrt(inputDim), 1/np.sqrt(inputDim), (hiddenDim, inputDim))
		V = np.random.uniform(-1/np.sqrt(hiddenDim), 1/np.sqrt(hiddenDim), (outputDim, hiddenDim))
		W = np.random.uniform(-1/np.sqrt(hiddenDim), 1/np.sqrt(hiddenDim), (hiddenDim, hiddenDim))
		self.U = theano.shared(name='U', value = U.astype(theano.config.floatX))
		self.V = theano.shared(name='V', value = V.astype(theano.config.floatX))
		self.W = theano.shared(name='W', value = W.astype(theano.config.floatX))
		self.bptt_truncate = bptt_trunc
		self.inputDim = inputDim
		self.outputDim = outputDim
		self.hiddenDim = hiddenDim
		self.Build()



	def Build(self):
		##Set up computation graphs
		U, V, W = self.U, self.V, self.W
		hiddenDim = self.hiddenDim
		def ff(xt, stprev, U, V, W):
			activation = U[:,xt] + W.dot(stprev)
			s = T.tanh(activation)
			o = T.nnet.softmax(V.dot(s))
			return [o[0],s]
		x = T.ivector()
		#O is a matrix
		[O,S], updates = theano.scan(ff, sequences=x, non_sequences=[U, V, W],
		 outputs_info=[None, dict(initial=np.zeros(hiddenDim))], strict=True,
		 truncate_gradient = self.bptt_truncate)

		y = T.ivector('y')
		Error = T.sum(T.nnet.categorical_crossentropy(O, y)) ##This works because y is one-hot encoding

		dU = T.grad(Error, U)
		dV = T.grad(Error, V)
		dW = T.grad(Error, W)


		##Def funcs
		self.FeedForward = theano.function([x], O)
		#self.BatchFeedforward = theano.function([Q], BatchO)
		self.CalcError = theano.function([x,y], Error)


		LR = T.dscalar()
		self.SGTrain = theano.function([x,y, LR], [],
			updates=[(U, U - LR*dU),
						(V, V - LR*dV),
						(W, W - LR*dW)])



		##Batch training
		##For some reason batch training is about 10x slower than regular training

		#Loop over rows in matrix and calculate summed error
		#Calculate gradients
		#Update weights
		# def Error(xt,yt):
		# 	[o, _], _ = theano.scan(ff, sequences=xt, non_sequences=[U, V, W],
		#  outputs_info=[None, dict(initial=np.zeros(hiddenDim))], strict=True,
		#  truncate_gradient = self.bptt_truncate)
		# 	error = T.sum(T.nnet.categorical_crossentropy(o,yt))
		# 	return [error, yt]

		# XBatch = T.imatrix()
		# YBatch = T.imatrix()
		# [bErrorVec, _], _ = theano.scan(Error, sequences=[XBatch, YBatch],
		# 	outputs_info=[None, None])

		# bError = T.sum(bErrorVec)
		# BdU = T.grad(bError,U)
		# BdV = T.grad(bError,V)
		# BdW = T.grad(bError,W)
		# BLR = T.dscalar()
		# self.BatchSGTrain = theano.function([XBatch, YBatch, BLR], [],
		# 	updates=[(U, U - BLR*BdU),
		# 				(V, V - BLR*BdV),
		# 				(W, W - BLR*BdW)])


	def calculate_total_loss(self, X, Y):
		return np.sum([self.CalcError(x,y) for x,y in zip(X,Y)])

	def GenerateSentence(self, word_to_index, index_to_word):
		wordsinsent = [word_to_index['SENTENCE_START']]
		while True:
			worddist = self.FeedForward(wordsinsent)
			chosenWord = self.ChooseWord(worddist[-1], word_to_index, index_to_word)
			wordsinsent.append(chosenWord)
			if (chosenWord == word_to_index['SENTENCE_END']):
				break
		
		convertedSent = [index_to_word[ind] for ind in wordsinsent]
		return convertedSent

	def ChooseWord(self, distribution, word_to_index, index_to_word):
		wordchosen = 0
		distribution = list(distribution)
		indofstar = word_to_index['*']
		del distribution[word_to_index['SENTENCE_START']]
		del distribution[indofstar-1]
		del distribution[word_to_index['UNKOWN_TOKEN']-2]
		distribution = distribution/np.sum(distribution)
		randvar = np.random.random()
		runningsum = 0
		for j in np.arange(len(distribution)):
			runningsum += distribution[j]
			if (randvar < runningsum):
				wordchosen = j
				break;
			
		if (wordchosen == 7998):
			wordchosen = 8000
		elif (wordchosen > indofstar-2):
			wordchosen += 2
		else:
			wordchosen += 1
		return wordchosen




if __name__ == '__main__':
	t = datetime.datetime.now().time()
	print('Program started at ', t)
	##Load training data
	filename = 'testsave.csv'
	with open(filename, 'r', newline='', encoding='utf-8') as f:
		reader = csv.reader(f)
		sentences = [s for s in reader]
	npzfile = np.load('wordtofromindex.npz')
	word_to_index = npzfile['wtoi'][()]
	index_to_word = npzfile['itow']
	word_dim = len(word_to_index)

	X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in sentences])
	Y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in sentences])

	##Make network
	t1 = time.time()
	NN = RNN(word_dim, word_dim)
	t2 = time.time()
	print('Build took ' + str(t2-t1) + ' seconds.')
	load_model_parameters('savedparameters.npz', NN)
	##Train

	###Print error before
	#print('Error before training: ', NN.calculate_total_loss(X_train, Y_train))
	###SG training
	
	if (len(sys.argv) > 1):
		if (sys.argv[1].lower() == 'train'):
			if (sys.argv[2] != None):
				numberoftrains = int(sys.argv[2])
			else:
				numberoftrains = 1000
			if (sys.argv[3] != None):
				learningrate = float(sys.argv[3])
			else:
				learningrate = 0.01

			indices = [np.random.randint(len(X_train)) for j in range(0, numberoftrains)]
			print('Training commenced.')
			t1 = time.time()
			for j in np.arange(numberoftrains):
				NN.SGTrain(X_train[indices[j]], Y_train[indices[j]], learningrate)
			t2 = time.time()
			print('Training took: ' + str(t2-t1) + ' seconds.')
			#print('Error after training: ', NN.calculate_total_loss(X_train, Y_train))
			###Print error aft
			save_model_parameters('savedparameters', NN)

		elif (sys.argv[1].lower() == 'speak'):
			numOfSents = 5
			if (len(sys.argv) > 2):
				if sys.argv[2] != None:
					numOfSents = int(sys.argv[2])
			for j in np.arange(numOfSents):
				sentence = NN.GenerateSentence(word_to_index, index_to_word)
				print(' '.join(sentence[1:-1]))
		elif (sys.argv[1].lower() == "calcerror"):
			if (sys.argv[2] != None):
				numberoftrains = int(sys.argv[2])
			else:
				numberoftrains = 1000
			indices = [np.random.randint(len(X_train)) for j in range(0, numberoftrains)]
			BatchX = [X_train[index] for index in indices]
			BatchY = [Y_train[index] for index in indices]
			print(NN.calculate_total_loss(BatchX, BatchY))
	timeend = datetime.datetime.now().time()
	print('Program terminated at ', timeend)
		# elif (sys.argv[1].lower() == 'batchtrain'):
		# 	if (sys.argv[2] != None):
		# 		numberoftrains = int(sys.argv[2])
		# 	else:
		# 		numberoftrains = 1000
		# 	if (sys.argv[3] != None):
		# 		learningrate = float(sys.argv[3])
		# 	else:
		# 		learningrate = 0.01
		# 	#Pick sentences to train on
		# 	indices = [np.random.randint(len(X_train)) for j in range(0, numberoftrains)]
		# 	preBatchX = [X_train[index] for index in indices]
		# 	preBatchY = [Y_train[index] for index in indices]
		# 	#Find max length and pad the short sentences with 'SENTENCE_END' tokens
		# 	maxLen = max([len(sent) for sent in X_train])
		# 	BatchX = [preBatchX[j] + [word_to_index['SENTENCE_END']]*(maxLen-len(preBatchX[j]))
		# 		for j in range(0, len(preBatchX))]
		# 	BatchY = [preBatchY[j] + [word_to_index['SENTENCE_END']]*(maxLen-len(preBatchY[j]))
		# 		for j in range(0, len(preBatchY))]
		# 	#BatchX = np.asarray(BatchX)
		# 	#BatchY = np.asarray(BatchY)
		# 	##TODO Sanity check
		# 	print('Batch training commenced.')
		# 	t1 = time.time()
		# 	NN.BatchSGTrain(BatchX, BatchY, learningrate)
		# 	t2 = time.time()
		# 	print('Training took: ' + str(t2-t1) + ' seconds.')
		# 	save_model_parameters('savedparameters', NN)



	##Generate sentences
	##Test WildML's speed