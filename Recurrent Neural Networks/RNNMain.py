import theano
import theano.tensor as T
from BasicRNNModule import HiddenLayer as HL
import numpy as np



word_dim = 8000
hidden_dim = 100

X = T.dmatrix() #Input elements
hiddenLayer = HL(word_dim, hidden_dim, X)
Vp = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
V = theano.shared(value = Vp.astype(theano.config.floatX)) #Lin transformation from hidden to pre-act output
s = hiddenLayer.OutputVar
o, updates = theano.scan(lambda svec: T.nnet.softmax(V.dot(svec)), sequences=s)
Y = T.dmatrix()
costfunc = T.sum(T.sqr(Y-o)) ##important to use build-in function to setup costfunction. Seems they have some
##optimization that is needed
funccostfunct = theano.function([X,Y], costfunc)
print('Got cost func')
hiddenLayer.GetGradient(costfunc, X, Y)
dV = T.grad(costfunc, V)
trainV = theano.function([X, Y], [], updates=[(V, V - 0.01*dV)])

feedforward = theano.function([X], o)

def TrainItAll(x, y, N):
	for j in np.arange(N):
		trainV(x,y)
		hiddenLayer.GDTrain(x,y)

atestinput = np.random.normal(-1,1, (10, word_dim))
atesttarget = np.random.uniform(-1,1, (10, word_dim))
print(atestinput)
print('##'*30)
print(feedforward(atestinput))
print('##'*30)
print('Error before training: ', funccostfunct(atestinput, atesttarget))
TrainItAll(atestinput, atesttarget, 100)
print('Error after training: ', funccostfunct(atestinput, atesttarget))

