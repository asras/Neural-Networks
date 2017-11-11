import pandas as pd
import numpy as np
import tensorflow as tf
from CNN import CNN
from CNN1 import CNN1
import sys
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

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
if (len(sys.argv) > 1 and sys.argv[1].lower() == "cnn1"):
	aCNN = CNN1(sess=sess, save_path = "./model/mnistmodel1.ckpt")
else:
	aCNN = CNN(sess=sess, save_path = "./model/mnistmodel.ckpt")
t2 = time.time()
print("Build took {} seconds.".format(t2-t1))
print("Beginning training.")
t0 = time.time()
for j in range(number_of_batches):

	X_batch, y_targets = mnist.train.next_batch(number_of_samples)
	X_batch = X_batch.reshape([X_batch.shape[0], 28, 28, 1])
	print("Starting batch {}/{}".format(j+1, number_of_batches), end="\r")
	loss = aCNN.train(sess, X_batch, y_targets)

print("Finished batch {}/{}".format(number_of_batches, number_of_batches))
t3 = time.time()
print("Training completed. Duration: {}. Final loss: {}.".format(t3-t0, loss))
aCNN.save_model(sess)


print("Validating.")

loss_val = aCNN.calculate_loss(sess,
 mnist.test.images.reshape(
	[mnist.test.images.shape[0], 28, 28, 1]
	),
	 mnist.test.labels
)[0]
print("Loss on validation set: {}.".format(loss_val))
accuracy = aCNN.calculate_accuracy(sess,
 mnist.test.images.reshape([mnist.test.images.shape[0], 28, 28, 1]),
  mnist.test.labels
)
print("Accuracy on validation set: {}".format(accuracy))





