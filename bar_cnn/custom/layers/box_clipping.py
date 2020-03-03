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


class BoxClipping(tf.keras.layers.Layer):
    """ Keras layer to clip box values to lie inside a given shape.

    Attributes:
        keras.layers.Layer: Base layer class.
            This is the class from which all layers inherit.
                -   A layer is a class implementing common neural networks
                    operations, such as convolution, batch norm, etc.
                -   These operations require managing weights,
                    losses, updates, and inter-layer connectivity.
    """

    def call(self, inputs, **kwargs):
        """TODO: docstring

        Args:
            inputs (TODO): TODO

        **kwargs:
            TODO

        Returns:
            TODO
        """

        image, boxes = inputs
        shape = tf.cast(tf.shape(image), tf.keras.backend.floatx())

        if tf.keras.backend.image_data_format() == 'channels_first':
            _, _, height, width = tf.unstack(value=shape, axis=0)
        else:
            _, height, width, _ = tf.unstack(value=shape, axis=0)

        x1, y1, x2, y2 = tf.unstack(value=boxes, axis=-1)
        x1 = tf.clip_by_value(x1, clip_value_min=0, clip_value_max=width)
        y1 = tf.clip_by_value(y1, clip_value_min=0, clip_value_max=height)
        x2 = tf.clip_by_value(x2, clip_value_min=0, clip_value_max=width)
        y2 = tf.clip_by_value(y2, clip_value_min=0, clip_value_max=height)

        return tf.stack(values=[x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        """TODO: docstring

        Args:
            input_shape (TODO): TODO

        Returns:
            TODO
        """
        return input_shape[1]
