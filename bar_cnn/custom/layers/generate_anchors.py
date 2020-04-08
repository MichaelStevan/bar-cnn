# Library Imports
import tensorflow as tf
import numpy as np

# Local Imports
from bar_cnn import utils


class GenerateAnchors(tf.keras.layers.Layer):
    """ tf.keras layer for generating anchors for a given shape.

    Attributes:
        tf.keras.layers.Layer: Base layer class.
            This is the class from which all layers inherit.
                -   A layer is a class implementing common neural networks
                    operations, such as convolution, batch norm, etc.
                -   These operations require managing weights,
                    losses, updates, and inter-layer connectivity.
    """

    def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):
        """ Constructor method for this class

        Instantiate the class with the given args.
        __init__ is called when ever an object of the class is constructed.

        Args:
            size (TODO): The base size of the anchors to generate.
            stride (TODO): The stride of the anchors to generate.
            ratios (TODO, optional): The ratios of the anchors to generate
            scales (TODO, optional): The scales of the anchors to generate
            *args: TODO
            **kwargs: TODO
        """
        self.size = size
        self.stride = stride

        anchor_parameters = utils.anchors.AnchorParameters(sizes=self.size, strides=self.stride,
                                                           ratios=ratios, scales=scales)

        self.ratios = anchor_parameters.ratios
        self.scales = anchor_parameters.scales

        self.num_anchors = anchor_parameters.num_anchors

        self.anchors = anchor_parameters.generate_anchors(base_size=self.size).astype(np.float32)
        super(GenerateAnchors, self).__init__(*args, **kwargs)

    # TODO: Understanding
    def call(self, inputs, **kwargs):
        """ TODO: docstring

        Args:
            inputs (TODO): TODO

        **kwargs:
            TODO

        Returns:
            TODO

        """
        features = inputs
        features_shape = tf.shape(features)

        # generate proposals from bbox deltas and shifted anchors
        if tf.keras.backend.image_data_format() == 'channels_first':
            anchors = utils.anchors.shift(shape=features_shape[2:4],
                                          stride=self.stride,
                                          anchors=self.anchors)
        else:
            anchors = utils.anchors.shift(shape=features_shape[1:3],
                                          stride=self.stride,
                                          anchors=self.anchors)

        anchors = tf.tile(input=tf.expand_dims(anchors, axis=0),
                          multiples=(features_shape[0], 1, 1))

        return anchors

    # TODO: Understanding
    def compute_output_shape(self, input_shape):
        """TODO: docstring

        Args:
            input_shape (TODO): TODO

        Returns:
            TODO
        """

        if None not in input_shape[1:]:
            if tf.keras.backend.image_data_format() == 'channels_first':
                total = np.prod(input_shape[2:4]) * self.num_anchors
            else:
                total = np.prod(input_shape[1:3]) * self.num_anchors

            return input_shape[0], total, 4
        else:
            return input_shape[0], None, 4

    # TODO: Understanding
    def get_config(self):
        """TODO: docstring

        Returns:
            TODO
        """

        config = super(GenerateAnchors, self).get_config()
        config.update({
            'size': self.size,
            'stride': self.stride,
            'ratios': self.ratios.tolist(),
            'scales': self.scales.tolist(),
        })

        return config
