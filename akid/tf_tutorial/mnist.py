"""Builds the MNIST network.

This network is a modified network that combine convolutional neural
network(convolutional.py) and MLP(mnist.py) in the tutorial.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

TensorFlow install instructions:
https://tensorflow.org/get_started/os_setup.html

MNIST tutorial:
https://tensorflow.org/tutorials/mnist/tf/index.html
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow.python.platform
import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
SEED = 66478  # Set to None for random seed.
WEIGHT_DECAY = 5e-4
NUM_TRAIN = 50000
NUM_EVAL = 10000


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.

    Returns:
        Variable Tensor
    """
    var = tf.Variable(
        tf.truncated_normal(shape, stddev=stddev, seed=SEED),
        name=name)
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
        x: Tensor
    Returns:
        nothing
    """
    tf.summary.histogram(x.op.name + '/activations', x)
    tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


def inference(images):
    """Build the MNIST model up to where it may be used for inference.

    Args:
        images: Images placeholder, from inputs().

    Returns:
        softmax_linear: Output tensor with the computed logits.
    """
    # Conv 1
    with tf.name_scope('conv1'):
        kernel = _variable_with_weight_decay("weights", [5, 5, 1, 32],
                                             0.1, WEIGHT_DECAY)
        bias = tf.Variable(tf.zeros([32]), name='biases')
        conv = tf.nn.conv2d(images, kernel,
                            strides=[1, 1, 1, 1], padding='SAME',
                            name='conv')
        fmap = tf.nn.bias_add(conv, bias, name='fmap')
        relu = tf.nn.relu(fmap, name='relu')
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name='pool')
        _activation_summary(pool)
    # Conv 2
    with tf.name_scope('conv2'):
        kernel = _variable_with_weight_decay("weights", [5, 5, 32, 64],
                                             0.1, WEIGHT_DECAY)
        bias = tf.Variable(tf.constant(0.1, shape=[64]), name='biases')
        conv = tf.nn.conv2d(pool, kernel,
                            strides=[1, 1, 1, 1], padding='SAME',
                            name='conv')
        fmap = tf.nn.bias_add(conv, bias, name='fmap')
        relu = tf.nn.relu(fmap, name='relu')
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name='pool')
        _activation_summary(pool)
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    flattened_pool = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    flattened_pool_shape = flattened_pool.get_shape().as_list()
    # FC1
    with tf.name_scope('fc1'):
        fc = _variable_with_weight_decay('fc1', [flattened_pool_shape[1], 512],
                                         0.1, WEIGHT_DECAY)
        bias = tf.Variable(tf.constant(0.1, shape=[512]), name='biases')
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(flattened_pool, fc) + bias,
                            name='hidden')
        _activation_summary(hidden)
    # FC2
    with tf.name_scope('fc2'):
        fc = _variable_with_weight_decay('fc2', [512, NUM_CLASSES],
                                         0.1, WEIGHT_DECAY)
        bias = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]),
                           name='biases')
        logit = tf.nn.relu(tf.matmul(hidden, fc) + bias, name='hidden')
        _activation_summary(logit)
    return logit


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].

    Returns:
        loss: Loss tensor of type float.
    """
    # Convert from sparse integer labels in the range [0, NUM_CLASSSES)
    # to 1-hot dense float vectors (that is we will have batch_size vectors,
    # each with NUM_CLASSES values, all of which are 0.0 except there will
    # be a 1.0 in the entry corresponding to the label).
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat(axis=1, values=[indices, labels])
    onehot_labels = tf.sparse_to_dense(
        concated, tf.stack([batch_size, NUM_CLASSES]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=onehot_labels,
                                                            name='xentropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the
    # weight decay terms (L2 loss).
    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss


def training(loss, BATCH_SIZE):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.

    Returns:
        train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar(loss.op.name, loss)
    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    step = tf.Variable(0, name='step', trainable=False)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        step * BATCH_SIZE,  # Current index into the dataset.
        NUM_TRAIN,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    # Create a variable to track the global step.
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=step)
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).

    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label's is was in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))
