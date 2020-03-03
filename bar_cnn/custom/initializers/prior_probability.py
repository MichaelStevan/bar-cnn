# Built-in Imports
import math

# Library Imports
import tensorflow as tf
import numpy as np


class PriorProbability(tf.keras.initializers.Initializer):
    """ Apply a prior probability to the weights.

    Attributes:
        keras.initializers.Initializer: TODO
    """

    def __init__(self, probability=0.01):
        """ Constructor method for this class

        Instantiate the class with the given args.
        __init__ is called when ever an object of the class is constructed.

        Args:
            probability (float, optional) : TODO (defaults to 0.01)

        """
        self.probability = probability

    def get_config(self):
        """ TODO: docstring

        Returns:
            TODO
        """
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        """ TODO: docstring

        Args:
            shape (TODO)           : TODO
            dtype (TODO, optional) : TODO (defaults to None)

        Returns:
            TODO
        """

        # set bias to -log((1 - p)/p) for foreground
        foreground_bias = -math.log((1 - self.probability) / self.probability)
        result = tf.ones(shape, dtype=dtype) * foreground_bias

        return result
