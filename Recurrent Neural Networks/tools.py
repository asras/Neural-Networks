import numpy as np


def softmax(x):
	xt = np.exp(x-np.max(x))
	return xt/np.sum(xt)

def save_model_parameters(outfile, model):
	U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
	np.savez(outfile, U=U, V=V, W=W)
	print('Saved model parameters to %s.' % outfile)


def load_model_parameters(path, model):
	npzfile = np.load(path)
	U, V, W = npzfile['U'], npzfile['V'], npzfile['W']
	#model.hidden_dim = U.shape[0]
	#model.word_dim = U.shape[1]
	model.U.set_value(U)
	model.V.set_value(V)
	model.W.set_value(W)
	print('Loaded model parameters.')