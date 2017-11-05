import pandas as pd
import numpy as np
import tensorflow as tf
from CNN import CNN
import time
##TODO This really needs cleanup
def get_train_data():

	df = pd.read_csv("fashion-mnist_train0.csv")
	df.drop(["Unnamed: 0"], 1, inplace = True)
	df_labels = df["label"]
	df.drop(["label"], 1, inplace=True)
	X_array = [df.ix[ind].values.reshape([28,28,1]) for ind in range(len(df.index))]
		
	y_targets_array = [df_labels.ix[ind] for ind in range(len(df_labels.index))]
		

	for j in range(1,4):
		df = pd.read_csv("fashion-mnist_train{}.csv".format(j))
		df.drop(["Unnamed: 0"], 1, inplace = True)
		df_labels = df["label"]
		df.drop(["label"], 1, inplace=True)
		X_tmp = [df.ix[ind].values.reshape([28,28,1]) for ind in range(len(df.index))]
		
		y_targets_tmp = [df_labels.ix[ind] for ind in range(len(df_labels.index))]
		
		X_array = X_array + X_tmp
		y_targets_array = y_targets_array + y_targets_tmp
	return np.array(X_array), np.array(y_targets_array)

def get_validation_data():
	df = pd.read_csv("fashion-mnist_train4.csv")
	df.drop(["Unnamed: 0"], 1, inplace = True)
	df_labels = df["label"]
	df.drop(["label"], 1, inplace=True)
	X_array = [df.ix[ind].values.reshape([28,28,1]) for ind in range(len(df.index))]
		
	y_targets_array = [df_labels.ix[ind] for ind in range(len(df_labels.index))]
		
	return np.array(X_array), np.array(y_targets_array)




X_batch, y_targets = get_train_data()


sess = tf.Session() ##TODO Should we close session? Google it
print("Building model.")
t1 = time.time()
aCNN = CNN(sess=sess)
t2 = time.time()
print("Build took {} seconds.".format(t2-t1))

print("Beginning training.")
t1 = time.time()
loss = aCNN.train(sess, X_batch, y_targets)
t2 = time.time()
print("Training completed. Duration: {}. Final loss: {}.".format(t2-t1, loss))
aCNN.save_model(sess)



X_batch_validation, y_targets_validation = get_validation_data()

loss_val = aCNN.calculate_loss(sess, X_batch_validation, y_targets_validation)
print("Loss on validation set: {}.".format(loss_val))






