import numpy as np
import theano.tensor as T
from theano import function
import theano

## Define symbols and a function
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x,y], z)
f2 = function([x,y], x+y)


print(f(2,3))
print('-----')
print(f2(2,3))
print('-'*20)
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x,y], z)
print(f([[1,2],[3,4]],[[10,20],[30,40]]))
print('-'*20)
print(f(np.array([[1,2],[3,4]]),np.array([[10,20],[30,40]])))


a = T.vector()
b = T.vector()
f = function([a,b], a**2+b**2+2*a*b)
f2 = function([a,b], a*b)
print(f([0,1],[0,2]))
print('-'*20)
print(f2([0,1],[3,0]))
print('-'*20)

##Element-wise sigmoid
x = T.dmatrix('x')
s = 1/(1+T.exp(-x))
logistic = function([x], s)
print(logistic([[0,1],[-1,-2]]))
print('-'*20)

##Function with multiple outputs
a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2
f = function([a,b], [diff, abs_diff, diff_squared])
print(f([[1,1],[1,1]], [[0,1],[2,3]]))
print('-'*20)

##Function with default value
x, y = T.dscalars('x', 'y')
z = x+y
f = function([x, theano.In(y, value=1)], z)
print(f(33))
print(f(33,2))
print('-'*20)


##Function with internal state
state = theano.shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state+inc)])
print(state.get_value())
accumulator(1)
print(state.get_value())
accumulator(300)
print(state.get_value())
state.set_value(-1)
accumulator(3)
print(state.get_value())
print('-'*20)

##Skip use of shared/any symbolic variable
fn = state*2 + inc
foo = T.scalar(dtype=state.dtype)
skip_shared = function([inc, foo], fn, givens=[(state,foo)])
print(skip_shared(1,3))
print('-'*20)


##Copy a function - "The optimized graph of the original fct is copied"
##Performance efficiency
new_state = theano.shared(0)
new_accumulator = accumulator.copy(swap={state:new_state})


##Random numbers in Theano
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=234)
rv_u = srng.uniform((2,2))
rv_n = srng.normal((2,2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates=True) ##doesn update rv_n
nearly_zeros = function([], rv_u + rv_u - 2*rv_u)
print(f())
print(f())
print(g())
print(g())
print(nearly_zeros()) ##Random numbers are drawn only once during any single
##function call
print('-'*20)



