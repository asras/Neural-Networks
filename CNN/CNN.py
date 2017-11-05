import numpy as np
import tensorflow as tf
import os


##How to build a network:
###1. Define the feedforward computations (/graph)
###2. Define the loss
###3. Define the training operations

##Remember to name layers to enable saving and restoring of model parameters

class CNN:

	def __init__(self, sess=None, save_path = "./model/model.ckpt"):
		#train_mode is needed for the dropout layer to tell it when to
		#use dropout
		self.train_mode = False 
		self.save_path = save_path if save_path.endswith(".ckpt") else save_path + ".ckpt"
		self._get_model(sess)
		self._build_loss()
		self._build_train_op()



	def _get_model(self, sess):
		self._build_model()
		#tf.train.Saver() is kind of weird, so we need to tack on .meta when
		#checking for saved parameters.
		if os.path.exists(self.save_path + ".meta"):
			self._restore_model(sess)
		else:
			self._init_model(sess)
		

	def _restore_model(self, sess):
		saver = tf.train.Saver()
		saver.restore(sess, self.save_path)
		print("Restored model from saved parameters.")


	def _init_model(self, sess):
		init = tf.global_variables_initializer()
		sess.run(init)
		print("Initialized model randomly.")


	def _build_model(self):
		#From stackoverflow:
		#-1 is a placeholder that says "adjust as necessary to match the size
		# needed for the full tensor." It's a way of making the code be 
		#independent of the input batch size, so that you can change your 
		#pipeline and not have to adjust the batch size everywhere in the code.

		#I think None serves the same purpose for placeholders
		self.input_layer = tf.placeholder(tf.float32, [None, 28, 28, 1])
		

		#maybe try strides in the conv layers
		conv1 = tf.layers.conv2d(
			inputs = self.input_layer,
			filters = 32,
			kernel_size = [5,5],
			padding="same",
			activation = tf.nn.relu,
			name = "conv1")

		#One can think of the result of a conv layer as a new image
		#with the same dims but n_filters new colors, corresponding to
		#the n_filters filters.

		conv12 = tf.layers.conv2d(
			inputs = conv1,
			filters = 32,
			kernel_size = [5,5],
			padding = "same",
			activation = tf.nn.relu,
			name = "conv12")

		pool1 = tf.layers.max_pooling2d(
			inputs=conv12,
			pool_size=[2,2],
			strides=2,
			name = "pool1")

		conv2 = tf.layers.conv2d(
			inputs = pool1,
			filters = 64,
			kernel_size = [5,5],
			padding = "same",
			activation = tf.nn.relu,
			name = "conv2")

		pool2 = tf.layers.max_pooling2d(inputs=conv2,
			pool_size=[2,2],
			strides=2,
			name = "pool2")

		#Two times pooling to get 7 x 7 and 64 filters to get 7*7*64 variables
		pool2_flat = tf.reshape(pool2, [-1, 7*7*64], name = "pool2_flat")

		dense = tf.layers.dense(inputs=pool2_flat,
		 	units=1024,
		 	activation=tf.nn.relu,
		 	name = "dense")

		dropout = tf.layers.dropout(inputs = dense,
			rate = 0.4,
			training = self.train_mode,
			name = "dropout")

		self.logits = tf.layers.dense(inputs=dropout,
			units=10,
			name = "logits")

		
	def _build_loss(self):
		self.target_labels = tf.placeholder(tf.int32, [None], name = "target_labels")
		onehot_labels = tf.one_hot(indices=self.target_labels, depth = 10, 
			name = "onehot_labels")
		self.loss = tf.losses.softmax_cross_entropy(
			onehot_labels = onehot_labels, logits = self.logits)

	def _build_train_op(self):
		optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
		#Optimizers vary their parameters based on number of batches seen
		#/global_step which is why we provide it
		self.train_op = optimizer.minimize(
			loss = self.loss,
			global_step = tf.train.get_global_step(),
			name = "train_op"
			)


	def train(self, sess, X, y_target):
		#We want to run loss minimization so we need input and target
		#feed dict

		#X can be a batch because of -1 in build
		feed_dict = {self.input_layer : X, self.target_labels: y_target}
		self.train_mode = True
		loss, _ = sess.run(
			[self.loss, self.train_op], feed_dict)
		self.train_mode = False
		return loss

	def calculate_loss(self, sess, X, y_target):

		feed_dict = {self.input_layer : X, self.target_labels: y_target}
		loss = sess.run([self.loss], feed_dict)
		return loss


	def save_model(self, sess):
		saver = tf.train.Saver()

		save_path = saver.save(sess, self.save_path)
		print("Model saved in {}.".format(save_path))


#Basic testing
if __name__ == "__main__":
	sess = tf.Session()
	aCNN = CNN(sess=sess, save_path="./model/testsave.ckpt")
	aCNN.save_model(sess)
