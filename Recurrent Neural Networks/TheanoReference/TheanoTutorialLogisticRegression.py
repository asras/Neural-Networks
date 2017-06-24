import numpy as np
import theano
import theano.tensor as T
rng = np.random

N = 400 #Training sample size
feats = 784 #Number of input variables

#Generate dataset D = (input_variables, target_class)
D = (rng.randn(N,feats), rng.randint(size=N, low=0, high=2)) #tuple
training_steps = 10000

x = T.dmatrix('x')
y = T.dvector('y')


# initialize the weight vector w randomly
#
# this and the following bias variable b
# are shared so they keep their values
# between training iterations (updates)
w = theano.shared(rng.rand(feats), name='w')
b = theano.shared(0., name='b')

print('Initial model:')
print(w.get_value())
print(b.get_value())


# Construct Theano expression graph
p_1 = 1/(1+T.exp(-T.dot(x,w)-b))
prediction = p_1 > 0.5

xent = -y*T.log(p_1)-(1-y)*T.log(1-p_1)
cost = xent.mean() + 0.01*(w**2).sum()
gw, gb = T.grad(cost, [w,b])




#Compile - Set up function graphs
train = theano.function(
	inputs=[x,y],
	##outputs=[prediction, xent], Not needed
	updates=((w,w-0.1*gw), (b, b-0.1*gb))) ##this some crazy shit
predict = theano.function(inputs=[x], outputs=prediction)

#Train
for i in range(training_steps):
	##pred, err = train(D[0], D[1])
	train(D[0], D[1])

#print("Final model:")
#print(w.get_value())
#print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))