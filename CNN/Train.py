import pandas as pd
import numpy as np
import tensorflow as tf
from CNN import CNN
import sys
import time
##TODO This really needs cleanup
def get_train_data(number_of_samples):
	set_to_use = np.random.randint(4)

	df = pd.read_csv("fashion-mnist_train{}.csv".format(set_to_use))
	df.drop(["Unnamed: 0"], 1, inplace = True)
	df_labels = df["label"]
	df.drop(["label"], 1, inplace=True)
	n_training_samples = np.min([len(df.index), number_of_samples])
	indices = np.random.choice(range(len(df.index)), n_training_samples,
		replace = False)

	X_array = np.array([df.ix[ind].values.reshape([28,28,1]) for ind in indices])
		
	y_targets_array = np.array([df_labels.ix[ind] for ind in indices])
	
	return np.array(X_array), np.array(y_targets_array)

def get_validation_data():
	df = pd.read_csv("fashion-mnist_train4.csv")
	df.drop(["Unnamed: 0"], 1, inplace = True)
	df_labels = df["label"]
	df.drop(["label"], 1, inplace=True)
	X_array = [df.ix[ind].values.reshape([28,28,1]) for ind in range(len(df.index))]
		
	y_targets_array = [df_labels.ix[ind] for ind in range(len(df_labels.index))]
		
	return np.array(X_array), np.array(y_targets_array)


if len(sys.argv) > 1:
	try:
		number_of_samples = int(sys.argv[1])
	except:
		print("Faulty input. Using default value.")
		number_of_samples = 10
else:
	number_of_samples = 10

print("Training on {} samples.".format(number_of_samples))
X_batch, y_targets = get_train_data(number_of_samples)


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

loss_val = aCNN.calculate_loss(sess, X_batch_validation, y_targets_validation)[0]
print("Loss on validation set: {}.".format(loss_val))
accuracy = aCNN.calculate_accuracy(sess, X_batch_validation, y_targets_validation)
print("Accuracy on validation set: {}".format(accuracy))






##TODO percentage correct