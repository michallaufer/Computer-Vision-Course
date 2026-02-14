
import numpy as np
import matplotlib.pyplot as plt
from icv83551.classifiers.fc_net import FullyConnectedNet
from icv83551.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from icv83551.solver import Solver

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def test_fc_net_batchnorm():
    print("Testing FullyConnectedNet with Batch Normalization")
    np.random.seed(231)
    N, D, H1, H2, C = 2, 15, 20, 30, 10
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=(N,))

    for reg in [0, 3.14]:
        print('Running check with reg = ', reg)
        model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                                  reg=reg, weight_scale=5e-2, dtype=np.float64,
                                  normalization='batchnorm')

        loss, grads = model.loss(X, y)
        print('Initial loss: ', loss)

        for name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
            print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
        if reg == 0: print()

def test_fc_net_layernorm():
    print("\nTesting FullyConnectedNet with Layer Normalization")
    np.random.seed(231)
    N, D, H1, H2, C = 2, 15, 20, 30, 10
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=(N,))

    for reg in [0, 3.14]:
        print('Running check with reg = ', reg)
        model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                                  reg=reg, weight_scale=5e-2, dtype=np.float64,
                                  normalization='layernorm')

        loss, grads = model.loss(X, y)
        print('Initial loss: ', loss)

        for name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
            print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
        if reg == 0: print()

def test_fc_net_dropout():
    print("\nTesting FullyConnectedNet with Dropout")
    np.random.seed(231)
    N, D, H1, H2, C = 2, 15, 20, 30, 10
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=(N,))

    for dropout_keep_ratio in [0.5, 0.75]:
        print('Running check with dropout = ', dropout_keep_ratio)
        model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                                  weight_scale=5e-2, dtype=np.float64,
                                  dropout_keep_ratio=dropout_keep_ratio, seed=123)

        loss, grads = model.loss(X, y)
        print('Initial loss: ', loss)

        # Relative errors should be around e-6 or less; Note that it's fine
        # if for dropout_keep_ratio=1 you have W2 error be on the order of e-5.
        for name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
            print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
        print()

if __name__ == '__main__':
    test_fc_net_batchnorm()
    test_fc_net_layernorm()
    test_fc_net_dropout()
