import numpy as np
from random import shuffle
#from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_dim = W.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    dw = np.zeros(W.shape) # 初始化每个sample 对每个参数的偏导 
    num_pos_margin = 0.0
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1.0 # note delta = 1
      if margin > 0:
        loss += margin
        dw[:,j] = X[i,:].T
        num_pos_margin += 1.0
    dw[:, y[i]] =  -num_pos_margin* X[i,:].T  #sample所属类 的参数偏导
    dW += dw
  
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.

  loss = loss*1./num_train
  # print("svm_loss_naive_part1:",loss)
  # print("regularization_part2:", 0.5*reg * np.sum(W * W))
  # Add regularization to the loss.
  loss += 0.5*reg * np.sum(W * W)

  dW = dW*1./num_train 
  
  #print("svm_naive_reg_part2:",reg*W )
  dW += reg*W 
                           
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW#, reg*W


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_dim = W.shape[0]

  scores = np.dot(X, W)
  
  scores_y = np.reshape(scores[range(num_train),y], (num_train,1))
  
  margins = np.maximum(np.zeros((num_train, num_classes)) , scores - scores_y + 1)
  margins[[range(num_train)],y] = 0
  
  loss = np.sum(margins)*1./num_train
  loss += 0.5*reg*np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pos_margin = (margins>0) * 1.0

  pos_margin_newShape = np.reshape(pos_margin, (num_train, num_classes, 1))
  X_newShape = np.reshape(X, (num_train, 1, num_dim))
  Gradients =  pos_margin_newShape * X_newShape
  sum_Gradients = np.sum(Gradients,0).T

  num_pos = np.reshape(np.sum(pos_margin,1),(-1,1))
  
  y_grad_values = -1.*(num_pos * X)
  y_label_pos = np.zeros((num_train,num_classes))
  y_label_pos[list(range(num_train)),y] = 1
  y_label_grad =np.reshape(y_grad_values,(num_train,1,num_dim)) * np.reshape(y_label_pos, (num_train,num_classes,1))
  sum_y_label_grad = np.sum(y_label_grad,0).T
 
  sum_Gradients += sum_y_label_grad
  

  avg_Gradients  = sum_Gradients*1./ num_train
  
  #print("svm_naive_reg_part2:",reg*W )
  
  dW = avg_Gradients + reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return loss, dW #, reg*W




