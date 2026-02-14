
import numpy as np
import matplotlib.pyplot as plt
from icv83551.layers import *
from icv83551.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def test_batchnorm_forward():
    print("Testing batchnorm_forward")
    np.random.seed(231)
    N, D1, D2, D3 = 200, 50, 60, 3
    X = np.random.randn(N, D1)
    W1 = np.random.randn(D1, D2)
    W2 = np.random.randn(D2, D3)
    a = np.maximum(0, X.dot(W1)).dot(W2)

    print('Before batch normalization:')
    print('  means: ', a.mean(axis=0))
    print('  stds: ', a.std(axis=0))

    # Means should be close to zero and stds close to one
    print('After batch normalization (gamma=1, beta=0)')
    a_norm, _ = batchnorm_forward(a, np.ones(D3), np.zeros(D3), {'mode': 'train'})
    print('  mean: ', a_norm.mean(axis=0))
    print('  std: ', a_norm.std(axis=0))

    # Now means should be close to beta and stds close to gamma
    gamma = np.asarray([1.0, 2.0, 3.0])
    beta = np.asarray([11.0, 12.0, 13.0])
    a_norm, _ = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
    print('After batch normalization (nontrivial gamma, beta)')
    print('  means: ', a_norm.mean(axis=0))
    print('  stds: ', a_norm.std(axis=0))

def test_batchnorm_backward():
    print("\nTesting batchnorm_backward")
    np.random.seed(231)
    N, D = 4, 5
    x = 5 * np.random.randn(N, D) + 12
    gamma = np.random.randn(D)
    beta = np.random.randn(D)
    dout = np.random.randn(N, D)

    bn_param = {'mode': 'train'}
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)

    dx_num = eval_numerical_gradient_array(lambda x: batchnorm_forward(x, gamma, beta, bn_param)[0], x, dout)
    da_num = eval_numerical_gradient_array(lambda gamma: batchnorm_forward(x, gamma, beta, bn_param)[0], gamma, dout)
    db_num = eval_numerical_gradient_array(lambda beta: batchnorm_forward(x, gamma, beta, bn_param)[0], beta, dout)

    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    print('dx error: ', rel_error(dx_num, dx))
    print('dgamma error: ', rel_error(da_num, dgamma))
    print('dbeta error: ', rel_error(db_num, dbeta))

def test_batchnorm_backward_alt():
    print("\nTesting batchnorm_backward_alt")
    np.random.seed(231)
    N, D = 100, 500
    x = 5 * np.random.randn(N, D) + 12
    gamma = np.random.randn(D)
    beta = np.random.randn(D)
    dout = np.random.randn(N, D)

    bn_param = {'mode': 'train'}
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)

    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    dx_alt, dgamma_alt, dbeta_alt = batchnorm_backward_alt(dout, cache)

    print('dx error: ', rel_error(dx, dx_alt))
    print('dgamma error: ', rel_error(dgamma, dgamma_alt))
    print('dbeta error: ', rel_error(dbeta, dbeta_alt))


if __name__ == '__main__':
    test_batchnorm_forward()
    test_batchnorm_backward()
    test_batchnorm_backward_alt()
