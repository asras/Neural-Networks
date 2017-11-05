import pickle
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from CNN import CNN
import time
#	df = pd.read_csv("sp500_joined_closes.csv")
df = pd.read_csv("fashion-mnist_train0.csv")
df.drop(["Unnamed: 0"], 1, inplace = True)
df_labels = df["label"]
df.drop(["label"], 1, inplace=True)

n_training_samples = len(df.index)
batch_size = 10000


batch = np.zeros([batch_size, 28, 28, 1]) #last index is number of color channels
random_indices = np.random.choice(range(n_training_samples),
 batch_size, replace=False)

X_batch = np.array([df.ix[ind].values.reshape([28,28,1]) for ind in random_indices])
y_targets = np.array([df_labels.ix[ind] for ind in random_indices])
sess = tf.Session()
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

#print(df.head())
#print(df.ix[0][:10])

# for j in range(5):
# 	imarray = np.zeros((28,28))
# 	im1 = df.ix[j]
# 	im1 = np.reshape(im1, (28,28))	
# 	plt.imshow(im1, cmap="gray", interpolation="none")
# 	plt.savefig("imtest{}.png".format(j))
# 	plt.clf()