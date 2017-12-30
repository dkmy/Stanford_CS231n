import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  for i in xrange(num_train):
    score = np.dot(X[i], W)
    score_exp = np.exp(score)
    score_exp_norm = score_exp / np.sum(score_exp)
    
    loss -= np.log(score_exp_norm[y[i]])
    
    score_exp_norm[y[i]] -= 1

    for s in xrange(num_classes):
      dW[:, s] += score_exp_norm[s] * X[i].T
    
  loss /= num_train
  dW /= num_train
  
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  #Score: N, C
  scores = np.dot(X, W)
  scores_exp = np.exp(scores)
  scores_exp /= np.sum(scores_exp, axis = 1).reshape(-1, 1)
  
  loss = -1 * np.sum(np.log(scores_exp[xrange(len(y)), y]))
  scores_exp[xrange(len(y)), y] -= 1
  
  dW = np.dot(X.T, scores_exp)
  

  loss /= num_train
  dW /= num_train
  
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  
  
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

