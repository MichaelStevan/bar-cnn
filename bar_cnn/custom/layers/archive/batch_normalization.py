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


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    Very similar to tf.keras.layers.BatchNormalization, with the additional option to freeze parameters.

    Attributes:
        tf.keras.layers.Layer: Base layer class.
            This is the class from which all layers inherit.
                -   A layer is a class implementing common neural networks
                    operations, such as convolution, batch norm, etc.
                -   These operations require managing weights,
                    losses, updates, and inter-layer connectivity.

    Traditional Definition:
        ```
            tf.keras.layers.BatchNormalization(
                axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                beta_initializer='zeros', gamma_initializer='ones',
                moving_mean_initializer='zeros', moving_variance_initializer='ones',
                beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
                fused=None, trainable=True, virtual_batch_size=None, adjustment=None, name=None,
                **kwargs
                )
        ```
    """

    def __init__(self, freeze, *args, **kwargs):
        """ Constructor method for this class

        Instantiate the class with the given args.
        __init__ is called when ever an object of the class is constructed.

        Args:
            freeze (boolean): Flag indicating whether or not to freeze parameters
            *args (variable length list of additional arguments): A NON-KEY-WORDED argument list
            **kwargs (variable length list of additional arguments):  A KEY-WORDED argument list
        """

        self.freeze = freeze
        super(BatchNormalization, self).__init__(*args, **kwargs)

        # set to non-trainable if freeze is true
        self.trainable = not self.freeze

    def call(self, *args, **kwargs):
        """ TODO: docstring

        Args:
            *args (variable length list of additional arguments): A NON-KEY-WORDED argument list
            **kwargs (variable length list of additional arguments):  A KEY-WORDED argument list

        Returns:
            TODO

        """

        # Force test mode if frozen, otherwise use default behaviour (i.e., training=None).
        if self.freeze:
            kwargs['training'] = False
        return super(BatchNormalization, self).call(*args, **kwargs)

    def get_config(self):
        """ TODO: docstring

        Returns:
            TODO
        """

        config = super(BatchNormalization, self).get_config()
        config.update({'freeze': self.freeze})
        return config
