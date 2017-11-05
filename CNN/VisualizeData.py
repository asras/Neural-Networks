import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("fashion-mnist_train0.csv")
df.drop(["Unnamed: 0"], 1, inplace=True)
print(df.head())


df.drop(["label"], 1, inplace = True)




for j in range(4):
	imarray = np.zeros((28,28))
	im1 = df.ix[j]
	im1 = np.reshape(im1, (28,28))
	plt.subplot(2,2, j+1)
	plt.imshow(im1, cmap="gray", interpolation="none")
	#plt.savefig("imtest{}.png".format(j))
	#plt.clf()

plt.show()