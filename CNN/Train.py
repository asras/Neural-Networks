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

	for j in range(1,4):
		df1 = pd.read_csv("fashion-mnist_train{}.csv".format(j))
		df1.drop(["Unnamed: 0"], 1, inplace = True)
		df_labels1 = df1["label"]
		df1.drop(["label"], 1, inplace=True)
		df = df.append(df1)
		df_labels = df_labels.append(df_labels1)
	return df, df_labels

def get_validation_data():
	df = pd.read_csv("fashion-mnist_train4.csv")
	df.drop(["Unnamed: 0"], 1, inplace = True)
	df_labels = df["label"]
	df.drop(["label"], 1, inplace=True)
	return df, df_labels




df, df_labels = get_train_data()

df_val, df_labels_val = get_validation_data()

n_training_samples = len(df.index)
batch_size = n_training_samples


batch = np.zeros([batch_size, 28, 28, 1]) #last index is number of color channels
random_indices = range(n_training_samples)

X_batch = np.array([df.ix[ind].values.reshape([28,28,1]) for ind in random_indices])
y_targets = np.array([df_labels.ix[ind] for ind in random_indices])
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


X_batch_validation = np.array([row.values.reshape([28,28,1]) for row in df_val.ix])
y_targets_validation = np.array([row for row in df_labels_val.ix])
loss_val = aCNN.calculate_loss(sess, X_batch_validation, y_targets_validation)
print("Loss on validation set: {}.".format(loss_val))

aCNN.save_model(sess)





