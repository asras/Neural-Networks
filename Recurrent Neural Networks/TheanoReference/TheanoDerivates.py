import numpy as np
import theano
import theano.tensor as T
from theano import pp
x = T.dscalar('x')
y = x**2
gy = T.grad(y,x)
print(pp(gy))
f = theano.function([x], gy)
print(f(4))
print(f.maker.fgraph.outputs[0])

print('-'*30)

x = T.dmatrix('x')
s = T.sum(1/(1+T.exp(-x)))
ffs = theano.function([x], s)
print(ffs([[1],[2]]))
gs = T.grad(s,x)
dlogistic = theano.function([x], gs)
print(dlogistic([[0,1],[-1,-2]]))


print('-'*30)

##Compute jacobian
x = T.dvector()
y = x**2
J, updates = theano.scan(lambda i, y, x : T.grad(y[i], x), 
	sequences=T.arange(y.shape[0]), non_sequences=[y,x])
f = theano.function([x], J)
print(f([4,4]))