import theano
import theano.tensor as T
import numpy as np
import time




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
		[O, _, _], _ = theano.scan(ff, sequences=x, non_sequences=[ Ux, Wx, Ui, Wi, Uf, Wf, Uo, Wo, V], outputs_info=[None, dict(initial=np.zeros(hiddenDim)), 
				dict(initial=np.zeros(hiddenDim))], strict=True)
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
		
		LR = T.dscalar()
		self.SGTrain = theano.function([x, y, LR], [], updates=[(Ux, Ux - dUx*LR), (Wx, Wx - dWx*LR), (Ui, Ui - dUi*LR), (Wi, Wi - dWi*LR),
					(Uf, Uf - dUf*LR), (Wf, Wf - dWf*LR), (Uo, Uo - dWo*LR), (Wo, Wo - dWf*LR), (V, V - dV*LR)])
		
		
		
		
		

		
		
if __name__ == '__main__':
	t1 = time.time()
	NN = LSTM(10, 10, 10)
	t2 = time.time()
	print('Build took ' + str(t2-t1) + ' seconds.')