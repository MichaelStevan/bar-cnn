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


class FilterDetections(tf.keras.layers.Layer):
    """ Keras layer for filtering detections

    Filters detections using score threshold, NMS and selecting the top-k detections.

    Attributes:
        keras.layers.Layer: Base layer class.
            This is the class from which all layers inherit.
                -   A layer is a class implementing common neural networks
                    operations, such as convolution, batch norm, etc.
                -   These operations require managing weights,
                    losses, updates, and inter-layer connectivity.
    """

    def __init__(self, nms=True, class_specific_filter=True,
                 nms_threshold=0.5, score_threshold=0.05,
                 max_detections=300, parallel_iterations=32,
                 **kwargs):
        """ Constructor method for this class

        Instantiate the class with the given args.
        __init__ is called when ever an object of the class is constructed.

        Args:
            nms (bool, optional): Flag to enable/disable NMS.
            class_specific_filter (bool, optional): Whether to perform filtering
                per class, or take the best scoring class and filter those.
            nms_threshold (float, optional): Threshold for the IoU value to determine
                when a box should be suppressed
            score_threshold (float, optional): Threshold used to pre-filter the boxes
            max_detections (int, optional): Maximum number of detections to keep
            parallel_iterations (int, optional): Number of batch items to process
                in parallel
            **kwargs:
                TBD

        Returns:
            TBD
        """

        self.nms = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.parallel_iterations = parallel_iterations
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ Constructs the NMS graph.

        Args:
            inputs (List of Tensors) : [boxes, classification, other[0], ...] tensors.
                -   boxes is shaped (max_detections, 4) and contains
                        the (x1, y1, x2, y2) of the non-suppressed boxes.
                -   scores is shaped (max_detections,) and contains
                        the scores of the predicted class.
                -   labels is shaped (max_detections,) and contains
                        the predicted label.
                -   other[i] is shaped (max_detections, ...) and contains
                        the filtered other[i] data.
        """
        boxes = inputs[0]
        classification = inputs[1]
        relationship = inputs[2]
        other = inputs[3:]
        print(inputs)

        # call filter_detections on each batch
        outputs = tf.map_fn(
            self.filter_detections,
            elems=[boxes, classification, relationship, other],
            dtype=[tf.keras.backend.floatx(),
                   tf.keras.backend.floatx(), 'int32',
                   tf.keras.backend.floatx(), 'int32'] +
                  [o.dtype for o in other],
            parallel_iterations=self.parallel_iterations
        )

        return outputs

    def filter_detections(self,
                          args):
        """ Filter detections using the boxes and classification values.

        Args:
            args (List of Tensors):
                - boxes (Tensor): Shape (num_boxes, 4) containing
                    the boxes in (x1, y1, x2, y2) format.
                - classification (Tensor): Shape (num_boxes, num_classes) containing
                    the classification scores.
                - relationship (Tensor): Shape (num_boxes, num_predicates) containing
                    the relationship scores.
                - other (List of Tensors, optional): Shape (num_boxes, ...) to filter
                    along with the boxes and classification scores.

        Returns:
            A list of [boxes, scores, labels, other[0], other[1], ...].
            boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
            scores is shaped (max_detections,) and contains the scores of the predicted class.
            labels is shaped (max_detections,) and contains the predicted label.
            other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
            In case there are less than max_detections detections, the tensors are padded with -1's.
        """

        boxes = args[0]
        classification = args[1]
        relationship = args[2]
        other = args[3]

        if other is None:
            other = []

        if self.class_specific_filter:
            all_indices = []

            # perform per class filtering
            print(classification.shape)
            for c in range(int(classification.shape[1])):
                scores = classification[:, c]
                labels = c * tf.ones((tf.shape(scores)[0],), dtype='int64')
                all_indices.append(self.sub_filter_fn(boxes, scores, labels))

            # concatenate indices to single tensor
            indices = tf.keras.backend.concatenate(tensors=all_indices, axis=0)

        else:
            scores = tf.keras.backend.max(classification, axis=1)
            labels = tf.keras.backend.argmax(classification, axis=1)
            indices = self.sub_filter_fn(boxes, scores, labels)

        # select top k
        scores = tf.gather_nd(classification, indices)
        labels = indices[:, 1]

        scores, top_indices = tf.nn.top_k(input=scores,
                                          k=tf.minimum(self.max_detections,
                                                       tf.shape(scores)[0]))

        # filter input using the final set of indices
        indices = tf.gather(indices[:, 0], top_indices)
        boxes = tf.gather(boxes, indices)
        labels = tf.gather(labels, top_indices)
        predicate_scores = tf.gather(relationship, indices)

        predicate_labels = tf.keras.backend.argmax(predicate_scores, axis=1)
        predicate_scores = tf.keras.backend.max(predicate_scores, axis=1)

        other_ = [tf.gather(o, indices) for o in other]

        # zero pad the outputs
        pad_size = \
            tf.keras.backend.maximum(0, self.max_detections - tf.shape(scores)[0])

        boxes = tf.pad(tensor=boxes,
                       paddings=[[0, pad_size], [0, 0]],
                       constant_values=-1)

        scores = tf.pad(tensor=scores,
                        paddings=[[0, pad_size]],
                        constant_values=-1)

        labels = tf.pad(tensor=labels,
                        paddings=[[0, pad_size]],
                        constant_values=-1)

        labels = tf.cast(labels, 'int32')

        predicate_scores = tf.pad(tensor=predicate_scores,
                                  paddings=[[0, pad_size]],
                                  constant_values=-1)

        predicate_labels = tf.pad(tensor=predicate_labels,
                                  paddings=[[0, pad_size]],
                                  constant_values=-1)

        predicate_labels = tf.cast(predicate_labels, 'int32')

        other_ = [tf.pad(tensor=o,
                         paddings=[[0, pad_size]] + [[0, 0] for _ in range(1, len(o.shape))],
                         constant_values=-1) for o in other_]

        # set shapes, since we know what they are
        boxes.set_shape([self.max_detections, 4])

        scores.set_shape([self.max_detections])
        labels.set_shape([self.max_detections])

        predicate_scores.set_shape([self.max_detections])
        predicate_labels.set_shape([self.max_detections])

        for o, shape in zip(other_, [list(tf.keras.backend.int_shape(o)) for o in other]):
            o.set_shape([self.max_detections] + shape[1:])

        return [boxes, scores, labels, predicate_scores, predicate_labels] + other_

    def sub_filter_fn(self, boxes, scores, labels):
        """TODO: docstring

        Args:
            boxes (TODO): TODO
            scores (TODO): TODO
            labels (TODO): TODO

        Returns:
            TODO

        """

        # threshold based on score
        indices = tf.where(tf.greater(scores, self.score_threshold))

        if self.nms:
            filtered_boxes = tf.gather_nd(boxes, indices)
            filtered_scores = tf.gather(scores, indices)[:, 0]

            # perform NMS
            nms_indices = tf.image.non_max_suppression(boxes=filtered_boxes,
                                                       scores=filtered_scores,
                                                       max_output_size=self.max_detections,
                                                       iou_threshold=self.nms_threshold)

            # filter indices based on NMS
            indices = tf.gather(indices, nms_indices)

        # add indices to list of all indices
        labels = tf.gather_nd(labels, indices)
        indices = tf.stack([indices[:, 0], labels], axis=1)

        return indices

    # TODO: Understanding
    def compute_output_shape(self, input_shape):
        """ Computes the output shapes given the input shapes.

        Args:
            input_shape (TBD): List of input shapes
                [boxes, classification,
                 relationship, other[0], ...].

        Returns:
            List of tuples representing the output shapes:
                [filtered_boxes.shape, filtered_scores.shape,
                 filtered_labels.shape, filtered_other[0].shape, ...]
        """

        return [
                   (input_shape[0][0], self.max_detections, 4),
                   (input_shape[1][0], self.max_detections),
                   (input_shape[1][0], self.max_detections),
                   (input_shape[2][0], self.max_detections),
                   (input_shape[2][0], self.max_detections),
               ] + [
                   tuple(
                       [input_shape[i][0],
                        self.max_detections] + list(input_shape[i][2:])
                   ) for i in range(3, len(input_shape))
               ]

    # TODO: Understanding
    # TODO: Better Commenting
    def compute_mask(self, inputs, mask=None):
        """ This is required in Keras when there is more than 1 output.

        Args:
            inputs (TBD) : TBD
            mask (TBD, optional) : TBD (defaults to None)

        """
        return (len(inputs) + 2) * [None]

    def get_config(self):
        """ Gets the configuration of this layer.

        Returns:
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections, self).get_config()
        config.update({
            'nms': self.nms,
            'class_specific_filter': self.class_specific_filter,
            'nms_threshold': self.nms_threshold,
            'score_threshold': self.score_threshold,
            'max_detections': self.max_detections,
            'parallel_iterations': self.parallel_iterations,
        })

        return config
