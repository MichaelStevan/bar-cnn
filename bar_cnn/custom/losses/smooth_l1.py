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
            y_true (Tensor): Ground truths from the generator of shape (B, N, 5).
                The last value for each box is the state of the anchor (one of : ignore, negative, positive).
            y_pred (Tensor): Ground truths from the network of shape (B, N, 4).

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
                     y=regression_diff - 0.5 / sigma_squared)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.math.maximum(1, tf.shape(indices)[0])
        normalizer = tf.cast(normalizer, dtype=tf.keras.backend.floatx())
        return tf.keras.backend.sum(regression_loss)/normalizer

    return _smooth_l1
