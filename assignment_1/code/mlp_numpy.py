"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.layers = []
    self.activations = []

    no_hidden_layers = len(n_hidden) #Total number of hidden layers

    #Connection from input to first hidden layer
    self.layers.append(LinearModule(n_inputs, n_hidden[0]))
    self.activations.append(ReLUModule())

    #Hidden layers
    for layer in range(0, no_hidden_layers-1):
      self.layers.append(LinearModule(n_hidden[layer], n_hidden[layer+1]))
      self.activations.append(ReLUModule())

    #Last hidden layer to output layer
    self.layers.append(LinearModule(n_hidden[-1], n_classes))
    self.activations.append(SoftMaxModule())

    self.loss_function = CrossEntropyModule()
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = x
    for idx, layer in enumerate(self.layers):
      out = layer.forward(out)
      out = self.activations[idx].forward(out)

    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    for layer_idx, layer in reversed(list(enumerate(self.layers))):
      dout = self.activations[layer_idx].backward(dout)
      dout =layer.backward(dout)

    ########################
    # END OF YOUR CODE    #
    #######################

    return
