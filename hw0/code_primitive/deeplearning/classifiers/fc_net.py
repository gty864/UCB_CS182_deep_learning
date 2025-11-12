import numpy as np

from deeplearning.layers import *
from deeplearning.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        self.reg = reg
        D, H, C = input_dim, hidden_dim, num_classes

        # Initialise weights and biases
        self.params = {
            'W1': np.random.randn(D, H) * weight_scale,
            'b1': np.zeros(H),
            'W2': np.random.randn(H, C) * weight_scale,
            'b2': np.zeros(C),
        }

    # ------------------------------------------------------------------
    # Loss / forward / backward
    # ------------------------------------------------------------------
    def loss(self, X, y=None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        reg = self.reg

        # ----- forward pass -----
        h1, cache1 = affine_forward(X, W1, b1)        # (N,H)
        h1_relu, cache_relu = relu_forward(h1)        # (N,H)
        scores, cache2 = affine_forward(h1_relu, W2, b2)  # (N,C)

        if y is None:            # test-time → just return scores
            return scores

        # ----- loss & softmax gradient -----
        loss, dscores = softmax_loss(scores, y)   # dscores shape (N,C)

        # ----- backward pass -----
        dx2, dW2, db2 = affine_backward(dscores, cache2)
        dx2 = relu_backward(dx2, cache_relu)
        dx1, dW1, db1 = affine_backward(dx2, cache1)

        # ----- L2 regularisation (factor 0.5 as requested) -----
        loss += 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2))
        dW1 += reg * W1
        dW2 += reg * W2

        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        return loss, grads


class FullyConnectedNet(object):
    """
    {affine - [BN] - relu - [dropout]} x (L-1) - affine - softmax
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # ----------------------------------------------------------
        # Initialise weights, biases, BN params
        # ----------------------------------------------------------
        dims = [input_dim] + hidden_dims + [num_classes]
        for l in range(1, self.num_layers + 1):
            self.params['W' + str(l)] = \
                np.random.randn(dims[l-1], dims[l]) * weight_scale
            self.params['b' + str(l)] = np.zeros(dims[l])

            if l < self.num_layers and self.use_batchnorm:
                self.params['gamma' + str(l)] = np.ones(dims[l])
                self.params['beta'  + str(l)] = np.zeros(dims[l])

        # dropout and BN config
        self.dropout_param = {'mode': 'train', 'p': dropout}
        if seed is not None:
            self.dropout_param['seed'] = seed

        self.bn_params = [{'mode': 'train'} for _ in range(self.num_layers - 1)]

        # cast to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    # ------------------------------------------------------------------
    # Loss: forward + backward
    # ------------------------------------------------------------------
    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # set run-time flags
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_p in self.bn_params:
                bn_p['mode'] = mode

        # ----------------------------------------------------------
        # Forward pass
        # ----------------------------------------------------------
        caches = []          # store every intermediate cache
        out = X
        for l in range(1, self.num_layers):
            # affine
            out, cache_affine = affine_forward(
                out,
                self.params['W' + str(l)],
                self.params['b' + str(l)])
            caches.append(cache_affine)

            # batch-norm
            if self.use_batchnorm:
                out, cache_bn = batchnorm_forward(
                    out,
                    self.params['gamma' + str(l)],
                    self.params['beta'  + str(l)],
                    self.bn_params[l-1])
                caches.append(cache_bn)

            # relu
            out, cache_relu = relu_forward(out)
            caches.append(cache_relu)

            # dropout
            if self.use_dropout:
                out, cache_do = dropout_forward(out, self.dropout_param)
                caches.append(cache_do)

        # final affine → scores
        scores, cache_last = affine_forward(
            out,
            self.params['W' + str(self.num_layers)],
            self.params['b' + str(self.num_layers)])
        caches.append(cache_last)

        if mode == 'test':
            return scores

        # ----------------------------------------------------------
        # Loss & softmax gradient
        # ----------------------------------------------------------
        loss, dscores = softmax_loss(scores, y)

        # add L2 regularisation (factor 0.5)
        reg_loss = 0.0
        for l in range(1, self.num_layers + 1):
            W = self.params['W' + str(l)]
            reg_loss += 0.5 * self.reg * np.sum(W * W)
        loss += reg_loss

        # ----------------------------------------------------------
        # Backward pass
        # ----------------------------------------------------------
        grads = {}

        # final affine
        dout, dw, db = affine_backward(dscores, caches.pop())
        grads['W' + str(self.num_layers)] = dw + self.reg * self.params['W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = db

        # propagate through hidden blocks
        for l in reversed(range(1, self.num_layers)):
            if self.use_dropout:
                dout = dropout_backward(dout, caches.pop())
            dout = relu_backward(dout, caches.pop())
            if self.use_batchnorm:
                dout, dgamma, dbeta = batchnorm_backward(dout, caches.pop())
                grads['gamma' + str(l)] = dgamma
                grads['beta'  + str(l)] = dbeta
            dout, dw, db = affine_backward(dout, caches.pop())
            grads['W' + str(l)] = dw + self.reg * self.params['W' + str(l)]
            grads['b' + str(l)] = db

        return loss, grads