from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        dimensions = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            self.params['W%d' % (i + 1)] = np.random.randn(dimensions[i], dimensions[i+1]) * weight_scale
            self.params['b%d' % (i + 1)] = np.zeros(dimensions[i+1])
            
            if self.normalization and i < self.num_layers - 1:
                self.params['gamma%d' % (i + 1)] = np.ones(dimensions[i+1])
                self.params['beta%d' % (i + 1)] = np.zeros(dimensions[i+1])

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        scores = None
        caches = {}
        layer_input = X
        
        for i in range(self.num_layers):
            W = self.params['W%d' % (i + 1)]
            b = self.params['b%d' % (i + 1)]
            
            # Affine
            out, cache_affine = affine_forward(layer_input, W, b)
            caches['affine%d' % (i + 1)] = cache_affine
            
            # Last layer has no Norm/ReLU/Dropout
            if i == self.num_layers - 1:
                scores = out
                break
                
            # Batch/Layer Norm
            if self.normalization:
                gamma = self.params['gamma%d' % (i + 1)]
                beta = self.params['beta%d' % (i + 1)]
                bn_param = self.bn_params[i]
                
                if self.normalization == 'batchnorm':
                    out, cache_norm = batchnorm_forward(out, gamma, beta, bn_param)
                elif self.normalization == 'layernorm':
                    out, cache_norm = layernorm_forward(out, gamma, beta, bn_param)
                caches['norm%d' % (i + 1)] = cache_norm
            
            # ReLU
            out, cache_relu = relu_forward(out)
            caches['relu%d' % (i + 1)] = cache_relu
            
            # Dropout
            if self.use_dropout:
                out, cache_dropout = dropout_forward(out, self.dropout_param)
                caches['dropout%d' % (i + 1)] = cache_dropout
            
            layer_input = out

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        loss, grads = 0.0, {}
        loss, dscores = softmax_loss(scores, y)
        
        # Add regularization to loss
        for i in range(self.num_layers):
            loss += 0.5 * self.reg * np.sum(self.params['W%d' % (i + 1)] ** 2)
            
        dout = dscores
        for i in range(self.num_layers - 1, -1, -1):
            # Last layer
            if i == self.num_layers - 1:
                dx, dw, db = affine_backward(dout, caches['affine%d' % (i + 1)])
                grads['W%d' % (i + 1)] = dw + self.reg * self.params['W%d' % (i + 1)]
                grads['b%d' % (i + 1)] = db
                dout = dx
                continue
                
            # Dropout
            if self.use_dropout:
                dout = dropout_backward(dout, caches['dropout%d' % (i + 1)])
                
            # ReLU
            dout = relu_backward(dout, caches['relu%d' % (i + 1)])
            
            # Norm
            if self.normalization:
                if self.normalization == 'batchnorm':
                    dout, dgamma, dbeta = batchnorm_backward(dout, caches['norm%d' % (i + 1)])
                elif self.normalization == 'layernorm':
                    dout, dgamma, dbeta = layernorm_backward(dout, caches['norm%d' % (i + 1)])
                grads['gamma%d' % (i + 1)] = dgamma
                grads['beta%d' % (i + 1)] = dbeta
                
            # Affine
            dx, dw, db = affine_backward(dout, caches['affine%d' % (i + 1)])
            grads['W%d' % (i + 1)] = dw + self.reg * self.params['W%d' % (i + 1)]
            grads['b%d' % (i + 1)] = db
            dout = dx

        return loss, grads
