
import numpy as np
from icv83551.layers import *
from icv83551.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def test_spatial_batchnorm():
    print("Testing spatial_batchnorm")
    np.random.seed(231)
    N, C, H, W = 2, 3, 4, 5
    x = 4 * np.random.randn(N, C, H, W) + 10

    print('Before spatial batch normalization:')
    print('  Shape: ', x.shape)
    print('  Means: ', x.mean(axis=(0, 2, 3)))
    print('  Stds: ', x.std(axis=(0, 2, 3)))

    # Means should be close to zero and stds close to one
    gamma, beta = np.ones(C), np.zeros(C)
    bn_param = {'mode': 'train'}
    out, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)
    print('After spatial batch normalization:')
    print('  Shape: ', out.shape)
    print('  Means: ', out.mean(axis=(0, 2, 3)))
    print('  Stds: ', out.std(axis=(0, 2, 3)))

    # Backward
    np.random.seed(231)
    N, C, H, W = 2, 3, 4, 5
    x = 5 * np.random.randn(N, C, H, W) + 12
    gamma = np.random.randn(C)
    beta = np.random.randn(C)
    dout = np.random.randn(N, C, H, W)

    bn_param = {'mode': 'train'}
    out, cache = spatial_batchnorm_forward(x, gamma, beta, bn_param)

    dx_num = eval_numerical_gradient_array(lambda x: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0], x, dout)
    da_num = eval_numerical_gradient_array(lambda gamma: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0], gamma, dout)
    db_num = eval_numerical_gradient_array(lambda beta: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0], beta, dout)

    dx, dgamma, dbeta = spatial_batchnorm_backward(dout, cache)
    print('dx error: ', rel_error(dx_num, dx))
    print('dgamma error: ', rel_error(da_num, dgamma))
    print('dbeta error: ', rel_error(db_num, dbeta))

def test_layernorm():
    print("\nTesting layernorm")
    np.random.seed(231)
    N, D = 4, 5
    x = 5 * np.random.randn(N, D) + 12
    gamma = np.random.randn(D)
    beta = np.random.randn(D)
    dout = np.random.randn(N, D)

    ln_param = {}
    out, cache = layernorm_forward(x, gamma, beta, ln_param)

    dx_num = eval_numerical_gradient_array(lambda x: layernorm_forward(x, gamma, beta, ln_param)[0], x, dout)
    da_num = eval_numerical_gradient_array(lambda gamma: layernorm_forward(x, gamma, beta, ln_param)[0], gamma, dout)
    db_num = eval_numerical_gradient_array(lambda beta: layernorm_forward(x, gamma, beta, ln_param)[0], beta, dout)

    dx, dgamma, dbeta = layernorm_backward(dout, cache)
    print('dx error: ', rel_error(dx_num, dx))
    print('dgamma error: ', rel_error(da_num, dgamma))
    print('dbeta error: ', rel_error(db_num, dbeta))

def test_spatial_groupnorm():
    print("\nTesting spatial_groupnorm")
    np.random.seed(231)
    N, C, H, W = 2, 6, 4, 5
    G = 2
    x = 4 * np.random.randn(N, C, H, W) + 10
    x_g = x.reshape((N * G, -1))
    print('Before spatial group normalization:')
    print('  Means: ', x_g.mean(axis=1))
    print('  Stds: ', x_g.std(axis=1))

    gamma, beta = np.ones((1, C, 1, 1)), np.zeros((1, C, 1, 1))
    gn_param = {'eps': 1e-5}
    out, _ = spatial_groupnorm_forward(x, gamma, beta, G, gn_param)
    
    out_g = out.reshape((N * G, -1))
    print('After spatial group normalization:')
    print('  Means: ', out_g.mean(axis=1))
    print('  Stds: ', out_g.std(axis=1))

    # Backward
    np.random.seed(231)
    N, C, H, W = 2, 6, 4, 5
    G = 2
    x = 5 * np.random.randn(N, C, H, W) + 12
    gamma = np.random.randn(1, C, 1, 1)
    beta = np.random.randn(1, C, 1, 1)
    dout = np.random.randn(N, C, H, W)

    gn_param = {}
    out, cache = spatial_groupnorm_forward(x, gamma, beta, G, gn_param)

    dx_num = eval_numerical_gradient_array(lambda x: spatial_groupnorm_forward(x, gamma, beta, G, gn_param)[0], x, dout)
    da_num = eval_numerical_gradient_array(lambda gamma: spatial_groupnorm_forward(x, gamma, beta, G, gn_param)[0], gamma, dout)
    db_num = eval_numerical_gradient_array(lambda beta: spatial_groupnorm_forward(x, gamma, beta, G, gn_param)[0], beta, dout)

    dx, dgamma, dbeta = spatial_groupnorm_backward(dout, cache)
    print('dx error: ', rel_error(dx_num, dx))
    print('dgamma error: ', rel_error(da_num, dgamma))
    print('dbeta error: ', rel_error(db_num, dbeta))

if __name__ == '__main__':
    test_spatial_batchnorm()
    test_layernorm()
    test_spatial_groupnorm()
