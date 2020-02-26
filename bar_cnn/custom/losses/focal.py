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

import tensorflow as tf


def focal(alpha=0.25, gamma=2.0):
    """ Create a functor for computing the focal loss.

    Args:
        alpha (float, optional): Scale the focal weight with alpha
        gamma (float, optional): Take the power of the focal weight with gamma

    Returns:
        A functor that computes the focal loss using the alpha and gamma.
    """

    def _focal(y_true, y_pred):
        """ Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args:
            y_true (Tensor): Target data from the generator with shape (B, N, num_classes).
            y_pred (Tensor): Predicted data from the network with shape (B, N, num_classes).

        Returns:
            The focal loss of y_pred w.r.t. y_true.
        """

        labels = y_true[:, :, :-1]

        # -1 for ignore, 0 for background, 1 for object
        anchor_state = y_true[:, :, -1]
        classification = y_pred

        # ###################################################################### #
        #                       filter out "ignore" anchors                      #
        # ###################################################################### #

        # ####################
        #
        # -- operation 1 --
        # this operation returns the coordinates of true elements of condition.
        # The coordinates are returned in a 2-D tensor where the first dimension (rows)
        # represents the number of true elements, and the second dimension (columns)
        # represents the coordinates of the true elements.
        #
        # Keep in mind, the shape of the output tensor can vary depending on how many
        # true values there are in input. Indices are output in row-major order.
        #
        # ####################
        #
        # -- operation 2 & 3 --
        # Gather slices from params into a Tensor with shape specified by indices.
        #
        # indices is an K-dimensional integer tensor,
        # best thought of as a (K-1)-dimensional tensor of indices into params,
        # where each element defines a slice of params.
        #
        # These slices are slices into the first N dimensions of params,
        # where N = indices.shape[-1]
        # ####################

        # operation 1
        indices = tf.where(condition=tf.math.not_equal(anchor_state, -1))

        # operation 2
        labels = tf.gather_nd(params=labels, indices=indices)

        # operation 3
        classification = tf.gather_nd(params=classification, indices=indices)
        # ###################################################################### #

        # compute the focal loss
        alpha_factor = tf.ones_like(input=labels)*alpha
        alpha_factor = tf.where(condition=tf.math.equal(labels, 1),
                                x=alpha_factor, y=1-alpha_factor)
        focal_weight = tf.where(condition=tf.math.equal(labels, 1),
                                x=1-classification, y=classification)
        focal_weight = alpha_factor*(focal_weight**gamma)

        cls_loss = \
            focal_weight * tf.keras.backend.binary_crossentropy(target=labels,
                                                                output=classification)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(condition=tf.math.equal(anchor_state, 1))
        normalizer = tf.cast(tf.shape(normalizer)[0], dtype=tf.keras.backend.floatx())
        normalizer = tf.math.maximum(tf.keras.backend.cast_to_floatx(1.0), normalizer)
        return tf.keras.backend.sum(cls_loss)/normalizer

    return _focal


# TODO: High Priority
# TODO: Refactoring
def smooth_l1(sigma=3.0):
    """ Create a smooth L1 loss functor.

    Args:
        sigma (float, optional): This argument defines the point where the loss changes from L2 to L1

    Returns:
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma**2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args:
            y_true (Tensor): Groundtruths from the generator of shape (B, N, 5).
                The last value for each box is the state of the anchor (one of : ignore, negative, positive).
            y_pred (Tensor): Groundtruths from the network of shape (B, N, 4).

        Returns:
            The smooth L1 loss of y_pred w.r.t. y_true.
        """

        # separate target and state
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices = tf.where(condition=tf.math.equal(anchor_state, 1))
        regression = tf.gather_nd(params=regression, indices=indices)
        regression_target = tf.gather_nd(params=regression_target, indices=indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = tf.math.abs(regression_diff)
        regression_loss = \
            tf.where(condition=tf.math.less(regression_diff, 1.0 / sigma_squared),
                     x=0.5*sigma_squared*tf.math.pow(regression_diff, 2),
                     y=regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = tf.math.maximum(1, tf.shape(indices)[0])
        normalizer = tf.cast(normalizer, dtype=tf.keras.backend.floatx())
        return tf.keras.backend.sum(regression_loss)/normalizer

    return _smooth_l1
