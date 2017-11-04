import pandas as pd


df = pd.read_csv("fashion-mnist_train.csv")

#df_new = df.ix[0:3]
#print(df_new.head())
batch_size = len(df.index)/6
for j in range(6):
	df_new = df.ix[j*batch_size:(j+1)*batch_size]
	df_new.to_csv("fashion-mnist_train{}.csv".format(j))
