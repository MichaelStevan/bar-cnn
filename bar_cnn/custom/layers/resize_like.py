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


class ResizeLike(tf.keras.layers.Layer):
    """ tf.keras layer to resize a tensor to the reference tensor shape.

    Attributes:
        keras.layers.Layer: Base layer class.
            This is the class from which all layers inherit.
                -   A layer is a class implementing common neural networks
                    operations, such as convolution, batch norm, etc.
                -   These operations require managing weights,
                    losses, updates, and inter-layer connectivity.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """TODO: docstring

        Args:
            inputs (TODO): TODO

        **kwargs:
            TODO

        Returns:
            TODO
        """

        input_tensor, ref_tensor = inputs

        return self.resize_like(input_tensor, ref_tensor)

    @staticmethod
    def resize_like(input_tensor, ref_tensor):
        """ Resize an image tensor to the same size/shape as a reference image tensor

        Args:
            input_tensor: (image tensor) Input image tensor that will be resized
            ref_tensor: (image tensor) Reference image tensor that we want to resize the input tensor to.

        Returns:
            reshaped tensor
        """
        reshaped_tensor = tf.image.resize(images=input_tensor,
                                          size=tf.shape(ref_tensor)[1:3],
                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                          preserve_aspect_ratio=False,
                                          antialias=False)
        return reshaped_tensor

    def compute_output_shape(self, input_shape):
        """TODO: docstring

        Args:
            input_shape (TODO): TODO

        Returns:
            TODO
        """
        if tf.keras.backend.image_data_format() == 'channels_first':
            return (input_shape[0][0], input_shape[0][1]) + input_shape[1][2:4]
        else:
            return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)
