"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data.
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module.

    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and
    std = 0.0001. Initialize biases self.params['bias'] with 0.

    Also, initialize gradients with zeros.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.params = {'weight': np.random.uniform(0, 0.0001, (out_features, in_features)), 'bias': np.zeros((out_features, 1))}
    self.grads = {'weight': np.zeros((out_features, in_features)), 'bias': np.zeros((out_features, 1))}
    self.x_l_minus_one = None
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.

    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.x_l_minus_one = x
    out = np.dot(self.params['weight'], x.T) + self.params['bias']
    ########################
    # END OF YOUR CODE    #
    #######################

    return out.T

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to
    layer parameters in self.grads['weight'] and self.grads['bias'].
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.grads['weight'] = np.dot(dout.T,self.x_l_minus_one)

    grad_b_shape = self.grads["bias"].shape
    self.grads["bias"] = np.reshape(dout.sum(axis=0), grad_b_shape)

    dx = np.dot(dout,self.params['weight'])
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """

  def __init__(self):
    self.relu = None
  def forward(self, x):
    """
    Forward pass.

    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = np.maximum(x, 0)
    self.relu = x
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = dout * (self.relu > 0)
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """

  def __init__(self):
    self.softmax = None
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    b = x.max(1)
    y = np.exp(np.subtract(x ,b.reshape(len(b),1)))
    out = y / y.sum(1)[:,None]
    self.softmax = out

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    batch_size = self.softmax.shape[0]  # batch size
    dim = self.softmax.shape[1]  # feature dimension

    diag_xN = np.zeros((batch_size, dim, dim))
    ii = np.arange(dim)
    diag_xN[:, ii, ii] = self.softmax

    dxdx_t = diag_xN - np.einsum('ij, ik -> ijk', self.softmax, self.softmax)
    dx = np.einsum('ij, ijk -> ik', dout, dxdx_t)
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss

    TODO:
    Implement forward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = ((-y * np.log(x)).sum(1)).mean()
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = -np.divide(y, x) / len(y)
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
