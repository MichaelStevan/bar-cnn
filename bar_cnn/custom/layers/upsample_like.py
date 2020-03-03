# Library imports
import tensorflow as tf


class UpsampleLike(tf.keras.layers.Layer):
    """ Keras layer for up-sampling a Tensor to be the same shape as another Tensor.

    Attributes:
        tf.keras.layers.Layer: Base layer class.
            This is the class from which all layers inherit.
                -   A layer is a class implementing common neural networks
                    operations, such as convolution, batch norm, etc.
                -   These operations require managing weights,
                    losses, updates, and inter-layer connectivity.
    """

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

        source, target = inputs
        target_shape = tf.shape(target)

        if tf.keras.backend.image_data_format() == 'channels_first':
            source = tf.transpose(source, perm=(0, 2, 3, 1))
            output = tf.image.resize(images=source,
                                     size=(target_shape[2], target_shape[3]),
                                     method='nearest')
            output = tf.transpose(output, perm=(0, 3, 1, 2))
            return output

        else:
            return tf.image.resize(images=source,
                                   size=(target_shape[1], target_shape[2]),
                                   method='nearest')

    # TODO: Understanding

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
