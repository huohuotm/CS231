import numpy as np
from random import shuffle
#from past.builtins import xrange

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
  #When you’re writing code for computing the Softmax function in practice, the intermediate terms efyiefyi and ∑jefj∑jefj may be very large due to the exponentials. 
  #Dividing large numbers can be numerically unstable, so it is important to use a normalization trick


  num_train = X.shape[0]
  num_dim = W.shape[0]
  num_class = W.shape[1]

  for i in range(num_train):
    score = np.dot(X[i,:], W)
    score_neg_max = -1.0*np.max(score)

    score_exp = np.exp(score+score_neg_max)
    score_exp_sum = np.sum(score_exp)
    y_score_exp = score_exp[y[i]]

    loss_i = -np.log(y_score_exp*1.0/score_exp_sum)
    loss += loss_i

    for j in range(num_class):
      dW[:,j] +=score_exp[j]*1.0/score_exp_sum * np.transpose(X[i,:])
    dW[:,y[i]] -= np.transpose(X[i,:])

  loss = loss*1.0/num_train + 0.5*reg*np.sum(W * W)
  dW = dW*1.0/num_train + reg*W
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
  num_train = X.shape[0]
  num_dim = W.shape[0]
  num_class = W.shape[1]

  scores = np.dot(X, W)
  scores_neg_max = -np.max(scores, 1)
  scores_exp = np.exp(scores+np.reshape(scores_neg_max,(-1,1)))
  scores_exp_sum = np.sum(scores_exp, 1)
  y_scores_exp = scores_exp[np.arange(num_train),y]

  loss = np.sum(-np.log(y_scores_exp*1./scores_exp_sum))/num_train + 0.5*reg*np.sum(W*W)

  grad = scores_exp*1./np.reshape(scores_exp_sum,(num_train,-1))
  grad[np.arange(num_train), y] -=1.0
  dW = np.dot(np.transpose(X), grad)*1.0/num_train + reg*W
  

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

