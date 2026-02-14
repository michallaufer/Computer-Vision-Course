
import numpy as np
from icv83551.layers import *
from icv83551.gradient_check import eval_numerical_gradient_array

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def test_dropout_forward():
    print("Testing dropout_forward")
    np.random.seed(231)
    x = np.random.randn(500, 500) + 10

    for p in [0.25, 0.4, 0.7]:
        out, _ = dropout_forward(x, {'mode': 'train', 'p': p})
        out_test, _ = dropout_forward(x, {'mode': 'test', 'p': p})

        print('Running tests with p = ', p)
        print('Mean of input: ', x.mean())
        print('Mean of train-time output: ', out.mean())
        print('Mean of test-time output: ', out_test.mean())
        print('Fraction of train-time output set to zero: ', (out == 0).mean())
        print('Fraction of test-time output set to zero: ', (out_test == 0).mean())

def test_dropout_backward():
    print("\nTesting dropout_backward")
    np.random.seed(231)
    x = np.random.randn(10, 10) + 10
    dout = np.random.randn(*x.shape)

    dropout_param = {'mode': 'train', 'p': 0.2, 'seed': 123}
    out, cache = dropout_forward(x, dropout_param)
    
    # We set seed so that the forward pass is deterministic
    dx = dropout_backward(dout, cache)
    
    # Numerically compute gradient. 
    # Since dropout is not differentiable at 0, we add a small perturbation
    # to avoid the non-differentiable points.
    
    dx_num = eval_numerical_gradient_array(lambda xx: dropout_forward(xx, dropout_param)[0], x, dout)

    print('dx relative error: ', rel_error(dx, dx_num))

if __name__ == '__main__':
    test_dropout_forward()
    test_dropout_backward()
