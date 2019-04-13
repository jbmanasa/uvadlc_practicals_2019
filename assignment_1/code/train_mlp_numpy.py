"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import cifar10_utils
# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  accuracy = (predictions.argmax(axis=1) == targets.argmax(axis=1)).mean()
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  #Parameters
  max_steps = FLAGS.max_steps
  batch_size = FLAGS.batch_size
  learning_rate = FLAGS.learning_rate
  eval_freq = FLAGS.eval_freq
  data_dir = FLAGS.data_dir

  #Get data
  data = cifar10_utils.get_cifar10(data_dir)
  train_data = data['train']
  validation_data = data['validation']
  test_data = data['test']

  #Get the shape of all data
  (_, channels, image_dim, _) = train_data.images.shape
  (_, mlp_classes) = train_data.labels.shape
  mlp_input_size = image_dim * image_dim * channels

  #Initialize the MLP
  NN = MLP(mlp_input_size, dnn_hidden_units, mlp_classes)

  for step in range(0,max_steps):
    x, y = train_data.next_batch(batch_size)
    x = np.reshape(x, [batch_size, mlp_input_size])

    out = NN.forward(x)
    loss = NN.loss_function.forward(out, y)
    dx = NN.loss_function.backward(out, y)
    NN.backward(dx)

    #Update weights
    for layer in NN.layers:
      layer.params['weight'] = layer.params['weight'] - learning_rate * layer.grads['weight']
      layer.params['bias']   = layer.params['bias'] - learning_rate * layer.grads['bias']

    if (step%eval_freq == 0) or step == max_steps-1:
      test_data_x, test_data_y = test_data.images.reshape((test_data.images.shape[0], mlp_input_size)), test_data.labels
      test_out = NN.forward(test_data_x)
      test_loss = NN.loss_function.forward(test_out, test_data_y)
      test_accuracy = accuracy(test_out, test_data_y)
      print("Step:", step, "Train Loss:", loss, "Test Loss:", test_loss, "Test Accuracy:", test_accuracy)

  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()