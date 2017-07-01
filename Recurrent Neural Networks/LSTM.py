import theano
import theano.tensor as T
import numpy as np
import time
import datetime
from tools import *
import csv
import sys



class LSTM(object):
	
	def __init__(self, inputDim, outputDim, hiddenDim, bptt_truncate=4):
		Ux = np.random.uniform(-1/np.sqrt(inputDim), 1/np.sqrt(inputDim), (hiddenDim, inputDim))
		Ui = np.random.uniform(-1/np.sqrt(inputDim), 1/np.sqrt(inputDim), (hiddenDim, inputDim))
		Uf = np.random.uniform(-1/np.sqrt(inputDim), 1/np.sqrt(inputDim), (hiddenDim, inputDim))
		Uo = np.random.uniform(-1/np.sqrt(inputDim), 1/np.sqrt(inputDim), (hiddenDim, inputDim))
		V = np.random.uniform(-1/np.sqrt(hiddenDim), 1/np.sqrt(hiddenDim), (outputDim, hiddenDim))
		Wx = np.random.uniform(-1/np.sqrt(hiddenDim), 1/np.sqrt(hiddenDim), (hiddenDim, hiddenDim))
		Wi = np.random.uniform(-1/np.sqrt(hiddenDim), 1/np.sqrt(hiddenDim), (hiddenDim, hiddenDim))
		Wf = np.random.uniform(-1/np.sqrt(hiddenDim), 1/np.sqrt(hiddenDim), (hiddenDim, hiddenDim))
		Wo = np.random.uniform(-1/np.sqrt(hiddenDim), 1/np.sqrt(hiddenDim), (hiddenDim, hiddenDim))
		print(theano.config.floatX)
		self.Ux = theano.shared(name='Ux', value=Ux.astype(theano.config.floatX))
		self.Ui = theano.shared(name='Ui', value=Ui.astype(theano.config.floatX))
		self.Uf = theano.shared(name='Uf', value=Uf.astype(theano.config.floatX))
		self.Uo = theano.shared(name='Uo', value=Uo.astype(theano.config.floatX))
		self.Wx = theano.shared(name='Wx', value=Wx.astype(theano.config.floatX))
		self.Wi = theano.shared(name='Wi', value=Wi.astype(theano.config.floatX))
		self.Wf = theano.shared(name='Wf', value=Wf.astype(theano.config.floatX))
		self.Wo = theano.shared(name='Wo', value=Wo.astype(theano.config.floatX))
		self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
		self.HiddenDim = hiddenDim
		self.Build()
		
	def Build(self):
		Ux, Wx, Ui, Wi, Uf, Wf, Uo, Wo, V = self.Ux, self.Wx, self.Ui, self.Wi, self.Uf, self.Wf, self.Uo, self.Wo, self.V
		hiddenDim = self.HiddenDim
		
		def ff(xt, st_prev, ct_prev, Ux, Wx, Ui, Wi, Uf, Wf, Uo, Wo, V):
			actX = Ux[:,xt] + Wx.dot(st_prev)
			g = T.tanh(actX)
			igate = Ui[:,xt] + Wi.dot(st_prev)
			fgate = Uf[:,xt] + Wf.dot(st_prev)
			ogate = Uo[:,xt] + Wo.dot(st_prev)
			ct = ct_prev*fgate + g*igate
			st = T.tanh(ct)*ogate
			ot = T.nnet.softmax(V.dot(st))
			return [ot[0], st, ct]
		
		x = T.ivector()		
		[O, S, C], updates = theano.scan(ff, sequences=x,
			outputs_info=[None, dict(initial=np.zeros(hiddenDim)), 
				dict(initial=np.zeros(hiddenDim))],
			 non_sequences= [ Ux, Wx, Ui, Wi, Uf, Wf, Uo, Wo, V],
			 strict=True)
		y = T.ivector()
		ErrorFunc = T.sum(T.nnet.categorical_crossentropy(O, y))
		
		dUx = T.grad(ErrorFunc, Ux)
		dWx = T.grad(ErrorFunc, Wx)
		dUi = T.grad(ErrorFunc, Ui)
		dWi = T.grad(ErrorFunc, Wi)
		dUf = T.grad(ErrorFunc, Uf)
		dWf = T.grad(ErrorFunc, Wf)
		dUo = T.grad(ErrorFunc, Uo)
		dWo = T.grad(ErrorFunc, Wo)
		dV = T.grad(ErrorFunc, V)
		
		self.FeedForward = theano.function([x], O)
		
		LR = T.scalar(dtype=theano.config.floatX)
		self.SGTrain = theano.function([x, y, LR], [], updates=[(Ux, Ux - LR*dUx),
		 (Wx, Wx - LR*dWx), (Ui, Ui - LR*dUi), (Wi, Wi - LR*dWi), 
		 (Uf, Uf - LR*dUf), (Wf, Wf - LR*dWf), (Uo, Uo - LR*dUo),
		  (Wo, Wo - LR*dWo), (V, V - LR*dV)])
		
		##Try training with sequential updating of parameters
		##Maybe this will allow for a higher learning rate
		
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
		indOfUnknownToken = word_to_index['UNKNOWN_TOKEN']
		del distribution[word_to_index['SENTENCE_START']]
		del distribution[word_to_index['UNKNOWN_TOKEN']-1]
		distribution = distribution/np.sum(distribution)
		randvar = np.random.random()
		runningsum = 0
		for j in np.arange(len(distribution)):
			runningsum += distribution[j]
			if (randvar < runningsum):
				wordchosen = j
				break;
			
		if (wordchosen > indOfUnknownToken - 2):
			wordchosen += 2
		else:
			wordchosen += 1
		return wordchosen
		

def OneHotsToMatrix(x, word_dim):
	X = np.zeros(len(x), word_dim)
	for j in range(len(x)):
		X[j, x[j]] = 1
	return X

	
		
if __name__ == '__main__':
	t = datetime.datetime.now().time()
	print('Program started at ', t)

	##Load training data
	filename = 'LSTM0.380451795826321160.028035619780282460.08099695271114749.csv'
	with open(filename, 'r', newline='', encoding='utf-8') as f:
		reader = csv.reader(f)
		sentences = [s for s in reader]
	npzfile = np.load('LSTMwordtofromindex.npz')	
	word_to_index = npzfile['wtoi'][()]
	index_to_word = npzfile['itow']
	word_dim = len(word_to_index)

	X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in sentences])
	Y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in sentences])

	#Make network
	t1 = time.time()
	NN = LSTM(inputDim=word_dim, outputDim=word_dim, hiddenDim=100)
	t2 = time.time()
	print('Build took ' + str(t2-t1) + ' seconds.')
	LSTM_load_model_parameters('LSTMsavedparameters.npz', NN)
	

	if (len(sys.argv) > 1):
		if (sys.argv[1].lower() == 'train'):
			if (sys.argv[2] != None):
				numberoftrains = int(sys.argv[2])
			else:
				numberoftrains = 1000
			if (sys.argv[3] != None):
				learningrate = float(sys.argv[3])
			else:
				learningrate = 0.001
			indices = [np.random.randint(len(X_train)) for j in range(0, numberoftrains)]
			print('Training commenced.')
			t1 = time.time()
			for j in np.arange(numberoftrains):
				NN.SGTrain(X_train[indices[j]], Y_train[indices[j]], learningrate)
			t2 = time.time()
			print('Training took: ' + str(t2-t1) + ' seconds.')
			#print('Error after training: ', NN.calculate_total_loss(X_train, Y_train))
			###Print error aft
			LSTM_save_model_parameters('LSTMsavedparameters', NN)

	