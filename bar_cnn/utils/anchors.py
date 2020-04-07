# Library Imports
import numpy as np
import tensorflow as tf

# Local Imports
from bar_cnn import utils


class AnchorParameters:
    """ TODO """

    def __init__(self, sizes=None, strides=None, ratios=None, scales=None):
        """ Constructor method for this class

        Instantiate the class with the given args.
        __init__ is called when ever an object of the class is constructed.

        Args:
            sizes (TODO, optional): List of sizes to use. Each size corresponds to one feature level.
            strides (TODO, optional): List of strides to use. Each stride correspond to one feature level.
            ratios (TODO, optional): List of ratios to use per location in a feature map.
            scales (TODO, optional): List of scales to use per location in a feature map.
        """
        self.defaults = {
            "sizes": [32, 64, 128, 256, 512],
            "strides": [8, 16, 32, 64, 128],
            "ratios": np.array([0.5, 1, 2],
                               dtype=tf.keras.backend.floatx()),
            "scales": np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
                               dtype=tf.keras.backend.floatx())
        }

        # If value is not defined assign default
        if sizes is not None:
            self.sizes = sizes
        else:
            self.sizes = self.defaults["sizes"]

        # If value is not defined assign default
        if strides is not None:
            self.strides = strides
        else:
            self.strides = self.defaults["strides"]

        # If value is not defined assign default
        if ratios is not None:
            self.ratios = ratios
        else:
            self.ratios = self.defaults["ratios"]

        # If value is not defined assign default
        if scales is not None:
            self.scales = scales
        else:
            self.scales = self.defaults["scales"]

        self.num_anchors = self.get_num_anchors()

    def get_num_anchors(self):
        """ TODO: docstring

        Returns:
            TODO
        """

        return len(self.ratios) * len(self.scales)

    def generate_anchors(self, base_size=16):
        """Generate anchor (reference) windows by enumerating aspect ratios X scales w.r.t. a reference window.

        Args:
            base_size (int, optional) : TODO (defaults to 16)

        Returns:
            TODO
        """

        # initialize output anchors
        anchors = np.zeros((self.num_anchors, 4))

        # scale base_size
        anchors[:, 2:] = base_size * np.tile(self.scales, (2, len(self.ratios))).T

        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]

        # correct for ratios
        anchors[:, 2] = np.sqrt(areas / np.repeat(self.ratios, len(self.scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(self.ratios, len(self.scales))

        # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

        return anchors


def shift(shape, stride, anchors):
    """Produce shifted anchors based on shape of the map and stride size.

    Args:
        shape (tuple): Shape to shift the anchors over.
        stride (TODO): Stride to shift the anchors with over the shape.
        anchors (TODO): The anchors to apply at each location.

    Returns:
        TODO
    """

    shift_x = \
        (tf.keras.backend.arange(start=0, stop=shape[1],
                                 dtype=tf.keras.backend.floatx()) +
         tf.constant(value=0.5, dtype=tf.keras.backend.floatx())) * stride

    shift_y = \
        (tf.keras.backend.arange(start=0, stop=shape[0],
                                 dtype=tf.keras.backend.floatx()) +
         tf.constant(value=0.5, dtype=tf.keras.backend.floatx())) * stride

    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = tf.reshape(shift_x, [-1])
    shift_y = tf.reshape(shift_y, [-1])

    shifts = tf.stack(values=[shift_x, shift_y, shift_x, shift_y], axis=0)
    shifts = tf.transpose(shifts)

    number_of_anchors = tf.shape(anchors)[0]

    # number of base points = feat_h * feat_w
    k = tf.shape(shifts)[0]

    shifted_anchors = \
        tf.reshape(anchors, shape=[1, number_of_anchors, 4]) + \
        tf.cast(tf.reshape(shifts, shape=[k, 1, 4]), tf.keras.backend.floatx())

    shifted_anchors = \
        tf.reshape(shifted_anchors, shape=[k*number_of_anchors, 4])

    return shifted_anchors


def bbox_transform_inv(boxes, deltas, mean=None, std=None):
    """Applies deltas (usually regression results) to boxes (usually anchors).

    Before applying the deltas to the boxes, the normalization that was
        previously applied (in the generator) has to be removed.
        The mean and std are the mean and std as applied in the generator.
        They are un-normalized in this function and then applied to the boxes.

    Args:
        boxes (np.array): shape (B, N, 4), where B is the batch size,
            N the number of boxes and 4 values for (x1, y1, x2, y2).
        deltas (np.array): same shape as boxes.
            These deltas (d_x1, d_y1, d_x2, d_y2) are a factor of the width/height.
        mean (np.array, optional): The mean value used when computing deltas
        std (np.array, optional): The standard deviation used when computing deltas

    Returns:
        A np.array of the same shape as boxes, but with deltas applied to each box.
        The mean and std are used during training to normalize the
            regression values (networks love normalization).
    """
    # Handle mean input
    if mean is None:
        mean = [0, 0, 0, 0]

    # Handle std input
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2]

    width = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]

    x1 = boxes[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
    y1 = boxes[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
    x2 = boxes[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
    y2 = boxes[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height

    pred_boxes = tf.stack(values=[x1, y1, x2, y2], axis=2)

    return pred_boxes
