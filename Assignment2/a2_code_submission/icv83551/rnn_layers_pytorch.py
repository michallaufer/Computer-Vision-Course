"""This file defines layer types that are commonly used for recurrent neural networks.
"""
import torch



def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A torch array containing input data, of shape (N, d_1, ..., d_k)
    - w: A torch array of weights, of shape (D, M)
    - b: A torch array of biases, of shape (M,)

    Returns:
    - out: output, of shape (N, M)
    """
    x_row = x.reshape(x.shape[0], -1)
    out = x_row.mm(w) + b
    return out


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """Run the forward pass for a single timestep of a vanilla RNN using tanh."""
    next_h = torch.tanh(x.mm(Wx) + prev_h.mm(Wh) + b)
    return next_h


def rnn_forward(x, h0, Wx, Wh, b):
    """Run a vanilla RNN forward on an entire sequence of data."""
    N, T, D = x.shape
    H = h0.shape[1]
    h = torch.zeros((N, T, H), dtype=x.dtype, device=x.device)
    prev_h = h0
    for t in range(T):
        prev_h = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)
        h[:, t, :] = prev_h
    return h


def word_embedding_forward(x, W):
    """Forward pass for word embeddings.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    """
    out = W[x]
    return out


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """Forward pass for a single timestep of an LSTM."""
    H = prev_h.shape[1]
    a = x.mm(Wx) + prev_h.mm(Wh) + b  # (N, 4H)
    ai, af, ao, ag = a[:, :H], a[:, H:2*H], a[:, 2*H:3*H], a[:, 3*H:]
    i = torch.sigmoid(ai)
    f = torch.sigmoid(af)
    o = torch.sigmoid(ao)
    g = torch.tanh(ag)
    next_c = f * prev_c + i * g
    next_h = o * torch.tanh(next_c)
    return next_h, next_c


def lstm_forward(x, h0, Wx, Wh, b):
    """Forward pass for an LSTM over an entire sequence of data."""
    N, T, D = x.shape
    H = h0.shape[1]
    h = torch.zeros((N, T, H), dtype=x.dtype, device=x.device)
    prev_h = h0
    prev_c = torch.zeros((N, H), dtype=x.dtype, device=x.device)
    for t in range(T):
        prev_h, prev_c = lstm_step_forward(x[:, t, :], prev_h, prev_c, Wx, Wh, b)
        h[:, t, :] = prev_h
    return h


def temporal_affine_forward(x, w, b):
    """Forward pass for a temporal affine layer.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns:
    - out: Output data of shape (N, T, M)
    """
    N, T, D = x.shape
    out = x.reshape(N * T, D).mm(w).reshape(N, T, -1) + b
    return out

def temporal_softmax_loss(x, y, mask, verbose=False):
    """A temporal version of softmax loss for use in RNNs.
    
    We assume that we are making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores for all vocabulary
    elements at all timesteps, and y gives the indices of the ground-truth element at each timestep.
    We use a cross-entropy loss at each timestep, summing the loss over all timesteps and averaging
    across the minibatch.

    As an additional complication, we may want to ignore the model output at some timesteps, since
    sequences of different length may have been combined into a minibatch and padded with NULL
    tokens. The optional mask argument tells us which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    loss = torch.nn.functional.cross_entropy(x_flat, y_flat, reduction='none')
    loss = loss * mask_flat.float()
    loss = loss.sum() / N

    return loss
