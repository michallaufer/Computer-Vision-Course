
import numpy as np
import matplotlib.pyplot as plt
from icv83551.classifiers.cnn import ThreeLayerConvNet
from icv83551.gradient_check import eval_numerical_gradient
from icv83551.fast_layers import *

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def test_sanity_loss():
    print("Testing Sanity Check Loss")
    model = ThreeLayerConvNet()
    N = 50
    X = np.random.randn(N, 3, 32, 32)
    y = np.random.randint(10, size=N)

    loss, grads = model.loss(X, y)
    print('Initial loss (no regularization): ', loss)

    model.reg = 0.5
    loss, grads = model.loss(X, y)
    print('Initial loss (with regularization): ', loss)
    # Expect ~2.3 for no reg.

def test_gradient_check():
    print("\nTesting Gradient Check")
    num_inputs = 2
    input_dim = (3, 16, 16)
    reg = 0.0
    num_classes = 10
    np.random.seed(231)
    X = np.random.randn(num_inputs, *input_dim)
    y = np.random.randint(num_classes, size=num_inputs)

    model = ThreeLayerConvNet(
        num_filters=3,
        filter_size=3,
        input_dim=input_dim,
        hidden_dim=7,
        dtype=np.float64
    )
    loss, grads = model.loss(X, y)
    # Errors should be small, but correct implementations may have
    # relative errors up to the order of e-2
    for param_name in sorted(grads):
        f = lambda _: model.loss(X, y)[0]
        param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)
        e = rel_error(param_grad_num, grads[param_name])
        print('%s max relative error: %e' % (param_name, e))

if __name__ == '__main__':
    test_sanity_loss()
    test_gradient_check()
