import pandas as pd
import numpy as np
import tensorflow as tf
from CNN import CNN
from CNN1 import CNN1
from CNN2 import CNN2
from CNN3 import CNN3
from CNN4 import CNN4
import sys
import time
##TODO This really needs cleanup
df_train = pd.read_csv("fashion-mnist_train{}.csv".format(0))
for j in range(1,4):
	df_train = df_train.append(pd.read_csv("fashion-mnist_train{}.csv".format(j)))
df_train.drop(["Unnamed: 0"], 1, inplace=True)
df_labels = df["label"]
df_train.drop(["label"], 1, inplace=True)

def get_train_data(number_of_samples):
	# set_to_use = np.random.randint(4)

	# df = pd.read_csv("fashion-mnist_train{}.csv".format(set_to_use))
	# df.drop(["Unnamed: 0"], 1, inplace = True)
	# df_labels = df["label"]
	# df.drop(["label"], 1, inplace=True)
	n_training_samples = np.min([len(df_train.index), number_of_samples])
	indices = np.random.choice(range(len(df_train.index)), n_training_samples,
		replace = False)

	X_array = np.array([df_train.ix[ind].values.reshape([28,28,1]) for ind in indices])
		
	y_targets_array = np.array([df_labels.ix[ind] for ind in indices])
	
	return np.array(X_array), np.array(y_targets_array)


def get_validation_data(number_of_samples):
	df = pd.read_csv("fashion-mnist_train4.csv")
	df.drop(["Unnamed: 0"], 1, inplace = True)
	df_labels = df["label"]
	df.drop(["label"], 1, inplace=True)
	n_training_samples = np.min([len(df.index), number_of_samples])
	indices = np.random.choice(range(len(df.index)), n_training_samples,
		replace = False)
	X_array = [df.ix[ind].values.reshape([28,28,1]) for ind in indices]
		
	y_targets_array = [df_labels.ix[ind] for ind in indices]
		
	return np.array(X_array), np.array(y_targets_array)


if len(sys.argv) > 2:
	try:
		number_of_batches = int(sys.argv[2])
	except:
		print("Faulty input. Using default value.")
		number_of_batches = 1
else:
	number_of_batches = 1


if len(sys.argv) > 3:
	try:
		number_of_samples = int(sys.argv[3])
	except:
		print("Faulty input. Using default value.")
		number_of_samples = 10
else:
	number_of_samples = 10


print("Performing training on {} batches of size {}.".format(number_of_batches, number_of_samples))
sess = tf.Session() ##TODO Should we close session? Google it
print("Building model.")
t1 = time.time()
#aCNN = ""
if (sys.argv[1].lower() == "cnn1"):
	aCNN = CNN1(sess=sess)
elif (sys.argv[1].lower() == "cnn2"):
	aCNN = CNN2(sess=sess)
elif (sys.argv[1].lower() == "cnn3"):
	aCNN = CNN3(sess=sess)
elif (sys.argv[1].lower() == "cnn4"):
	aCNN = CNN4(sess=sess)
else:
	aCNN = CNN(sess=sess)
t2 = time.time()
print("Build took {} seconds.".format(t2-t1))
print("Beginning training.")
t0 = time.time()
for j in range(number_of_batches):

	X_batch, y_targets = get_train_data(number_of_samples)
	print("Starting batch {}/{}".format(j+1, number_of_batches), end="\r")
	loss = aCNN.train(sess, X_batch, y_targets)

print("Finished batch {}/{}".format(number_of_batches, number_of_batches))
t3 = time.time()
print("Training completed. Duration: {}. Final loss: {}.".format(t3-t0, loss))
aCNN.save_model(sess)

##thea

X_batch_validation, y_targets_validation = get_validation_data(number_of_samples)
print("Validating on {} samples.".format(len(y_targets_validation)))

loss_val = aCNN.calculate_loss(sess, X_batch_validation, y_targets_validation)[0]
print("Loss on validation set: {}.".format(loss_val))
accuracy = aCNN.calculate_accuracy(sess, X_batch_validation, y_targets_validation)
print("Accuracy on validation set: {}".format(accuracy))





