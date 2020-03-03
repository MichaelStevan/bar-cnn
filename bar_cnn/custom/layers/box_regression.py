"""
Copyright 2019-2020 Darien Schettler (https://github.com/darien-schettler)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# PyPi Imports
import tensorflow as tf
import numpy as np

# Local Imports
from bar_cnn import utils


class BoxRegression(tf.keras.layers.Layer):
    """ Keras layer for applying regression values to boxes.

    Attributes:
        keras.layers.Layer: Base layer class.
            This is the class from which all layers inherit.
                -   A layer is a class implementing common neural networks
                    operations, such as convolution, batch norm, etc.
                -   These operations require managing weights,
                    losses, updates, and inter-layer connectivity.
    """

    def __init__(self, mean=None, std=None, **kwargs):
        """ Initializer for the BoxRegression layer.

        Args:
            mean (TODO, optional): The mean value of the regression values
                which was used for normalization.
            std (TODO, optional): The standard value of the regression values
                which was used for normalization. (defaults to None)
            **kwargs: TODO

        """
        # Handle input for mean
        if mean is None:
            self.mean = np.array([0, 0, 0, 0])
        else:
            if isinstance(mean, (list, tuple)):
                self.mean = np.array(mean)
            elif not isinstance(mean, np.ndarray):
                raise ValueError('Expected mean to be a np.ndarray, '
                                 'list or tuple. Received: {}'.format(type(mean)))

        # Handle input for std
        if std is None:
            self.std = np.array([0.2, 0.2, 0.2, 0.2])
        else:
            if isinstance(std, (list, tuple)):
                self.std = np.array(std)
            elif not isinstance(std, np.ndarray):
                raise ValueError('Expected std to be a np.ndarray, '
                                 'list or tuple. Received: {}'.format(type(std)))

        # Call to super init
        super().__init__(**kwargs)

    # TODO: Understanding
    def call(self, inputs, **kwargs):
        """TODO: docstring

        Args:
            inputs (TODO): TODO

        **kwargs:
            TODO

        Returns:
            TODO
        """
        anchors, regression = inputs
        return utils.anchors.bbox_transform_inv(boxes=anchors,
                                                deltas=regression,
                                                mean=self.mean,
                                                std=self.std)

    # TODO: Understanding
    def compute_output_shape(self, input_shape):
        """TODO: docstring

        Args:
            input_shape (TODO): TODO

        Returns:
            TODO
        """
        return input_shape[0]

    # TODO: Understanding
    def get_config(self):
        """TODO: docstring

        Returns:
            TODO
        """

        config = super(BoxRegression, self).get_config()
        config.update({
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
        })

        return config
