import numpy as np
import csv

def load_training_data(filename, dictfilename):
	with open(filename, 'r', newline='', encoding='utf-8') as f:
		reader = csv.reader(f)
		sentences = [s for s in reader]
	npzfile = np.load(dictfilename)
	word_to_index = npzfile['wtoi'][()]
	index_to_word = npzfile['itow']
	X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in sentences])
	Y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in sentences])
	return [word_to_index, index_to_word, X_train, Y_train]

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

def LSTM_save_model_parameters(outfile, model):
	Ux, Wx, Ui, Wi, Uf, Wf, Uo, Wo, V = model.Ux.get_value(), model.Wx.get_value(), model.Ui.get_value(), model.Wi.get_value(),	model.Uf.get_value(), model.Wf.get_value(),	model.Uo.get_value(), model.Wo.get_value(),	model.V.get_value()
	np.savez(outfile, Ux=Ux, Wx=Wx,
						Ui=Ui, Wi=Wi,
						Uf=Uf, Wf=Wf,
						Uo=Uo, Wo=Wo,
						V=V)
	print('Saved model parameters to %s.' % outfile)


def LSTM_load_model_parameters(path, model):
	npzfile = np.load(path)
	Ux, Wx, Ui, Wi, Uf, Wf, Uo, Wo, V = npzfile['Ux'], npzfile['Wx'], npzfile['Ui'], npzfile['Wi'],	npzfile['Uf'], npzfile['Wf'], npzfile['Uo'], npzfile['Wo'],	npzfile['V']
	#model.hidden_dim = U.shape[0]
	#model.word_dim = U.shape[1]
	model.Ux.set_value(Ux)
	model.Wx.set_value(Wx)
	model.Ui.set_value(Ui)
	model.Wi.set_value(Wi)
	model.Uf.set_value(Uf)
	model.Wf.set_value(Wf)
	model.Uo.set_value(Uo)
	model.Wo.set_value(Wo)
	model.V.set_value(V)
	print('Loaded model parameters.')


def Numpy_save_model_parameters(outfile, model):
	U, V, W = model.U, model.V, model.W
	np.savez(outfile, U=U, V=V, W=W)
	print('Saved model parameters to %s.' % outfile)


def Numpy_load_model_parameters(path, model):
	npzfile = np.load(path)
	U, V, W = npzfile['U'], npzfile['V'], npzfile['W']
	#model.hidden_dim = U.shape[0]
	#model.word_dim = U.shape[1]
	model.U = U
	model.V = V
	model.W = W
	print('Loaded model parameters.')