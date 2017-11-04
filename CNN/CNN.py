import numpy as np
import tensorflow as tf



class CNN:

	def __init__(self):
		self.train_mode = False
		self._build_model()
		self._build_loss()
		self._build_train_op()



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
			activation = tf.nn.relu)

		#One can think of the result of a conv layer as a new image
		#with the same dims but n_filters new colors, corresponding to
		#the n_filters filters.

		conv12 = tf.layers.conv2d(
			inputs = conv1,
			filters = 32,
			kernel_size = [5,5],
			padding = "same",
			activation = tf.nn.relu)

		pool1 = tf.layers.max_pooling2d(inputs=conv12, pool_size=[2,2], strides=2)

		conv2 = tf.layers.conv2d(
			inputs = pool1,
			filters = 64,
			kernel_size = [5,5],
			padding = "same",
			activation = tf.nn.relu)

		pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

		#Two times pooling to get 7 x 7 and 64 filters to get 7*7*64 vars
		pool2_flat = tf.reshape(pool2, [-1, 7*7*64])

		dense = tf.layers.dense(inputs=pool2_flat,
		 units=1024, activation=tf.nn.relu)
		dropout = tf.layers.dropout(inputs = dense, rate = 0.4,
			training = self.train_mode)

		self.logits = tf.layers.dense(inputs=dropout, units=10)

		# predictions = {
		# "classes" : tf.argmax(input=logits, axis=1),
		# "probabilities" : tf.nn.softmax(logits, name="softmax_tensor")
		# }

		# if mode == tf.estimator.ModeKeys.PREDICT:
		# 	return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

		# onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
		# loss = tf.losses.softmax_cross_entropy(
		# 	onehot_labels=onehot_labels, logits=logits)

		# if mode == tf.estimator.ModeKeys


	def _build_loss(self):
		self.target_labels = tf.placeholder(tf.int32, [None])
		onehot_labels = tf.one_hot(indices=self.target_labels, depth = 10)
		self.loss = tf.losses.softmax_cross_entropy(
			onehot_labels = onehot_labels, logits = self.logits)

	def _build_train_op(self):
		optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
		#Optimizers vary their parameters based on number of batches seen
		#/global_step which is why we provide it
		self.train_op = optimizer.minimize(
			loss = self.loss,
			global_step = tf.train.get_global_step()
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



aCNN = CNN()
