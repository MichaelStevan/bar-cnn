import tensorflow as tf


def filter_detections(
        boxes,
        classification,
        relationship,
        other=None,
        class_specific_filter=True,
        nms=True,
        score_threshold=0.05,
        max_detections=300,
        nms_threshold=0.5
):
    """ Filter detections using the boxes and classification values.
    Args
        boxes (tensor): Shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification (tensor): Shape (num_boxes, num_classes) containing the classification scores.
        other (list of tensors, optional): Used to filter along with the boxes and classification scores.
        class_specific_filter (bool, optional): Whether to perform filtering per class,
            or take the best scoring class and filter those.
        nms (bool, optional): Flag to enable/disable non maximum suppression.
        score_threshold (float, optional): Threshold used to prefilter the boxes with.
        max_detections (int, optional): Maximum number of detections to keep.
        nms_threshold (float, optional): Threshold for the IoU value to determine when a box should be suppressed.
    Returns
        A list of [boxes, scores, labels, other[0], other[1], ...].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """
    # Avoid `default argument is mutable` warning
    if other is None:
        other = []

    def _filter_detections(_scores, _labels):
        # Threshold based on score.
        _indices = tf.where(tf.keras.backend.greater(_scores, score_threshold))

        if nms:
            filtered_boxes = tf.gather_nd(boxes, _indices)
            filtered_scores = tf.keras.backend.gather(_scores, _indices)[:, 0]

            # Perform NMS.
            nms_indices = tf.image.non_max_suppression(filtered_boxes,
                                                       filtered_scores,
                                                       max_output_size=max_detections,
                                                       iou_threshold=nms_threshold)

            # Filter indices based on NMS.
            _indices = tf.keras.backend.gather(_indices, nms_indices)

        # Add indices to list of all indices.
        _labels = tf.gather_nd(_labels, _indices)
        _indices = tf.keras.backend.stack([_indices[:, 0], _labels], axis=1)

        return _indices

    if class_specific_filter:
        all_indices = []
        # Perform per class filtering.
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * tf.ones((tf.keras.backend.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))

        # Concatenate indices to single tensor.
        indices = tf.keras.backend.concatenate(all_indices, axis=0)
    else:
        scores = tf.keras.backend.max(classification, axis=1)
        labels = tf.keras.backend.argmax(classification, axis=1)
        indices = _filter_detections(scores, labels)

    # Select top k.
    scores = tf.gather_nd(classification, indices)
    labels = indices[:, 1]
    scores, top_indices = tf.nn.top_k(scores,
                                      k=tf.keras.backend.minimum(max_detections, tf.keras.backend.shape(scores)[0]))

    # Stop gradients. We do not want to train lower layers.
    scores = tf.stop_gradient(scores)
    top_indices = tf.stop_gradient(top_indices)

    # Filter input using the final set of indices.
    indices = tf.keras.backend.gather(indices[:, 0], top_indices)
    boxes = tf.keras.backend.gather(boxes, indices)
    labels = tf.keras.backend.gather(labels, top_indices)

    predicate_scores = keras.backend.gather(relationship, indices)
    predicate_labels = keras.backend.argmax(predicate_scores, axis=1)
    predicate_scores = keras.backend.max(predicate_scores, axis=1)

    other_ = [tf.keras.backend.gather(o, indices) for o in other]

    # Zero pad the outputs.
    pad_size = tf.keras.backend.maximum(0, max_detections - tf.keras.backend.shape(scores)[0])
    boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
    labels = tf.pad(labels, [[0, pad_size]], constant_values=-1)
    labels = tf.keras.backend.cast(labels, 'int32')

    predicate_scores = backend.pad(
        predicate_scores, [[0, pad_size]], constant_values=-1)
    predicate_labels = backend.pad(
        predicate_labels, [[0, pad_size]], constant_values=-1)
    predicate_labels = tf.keras.backend.cast(predicate_labels, 'int32')

    other_ = [tf.pad(o, [[0, pad_size]] + [[0, 0] for _ in range(1, len(o.shape))], constant_values=-1) for o in other_]

    # Set shapes, since we know what they are.
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])

    predicate_scores.set_shape([max_detections])
    predicate_labels.set_shape([max_detections])

    for o, s in zip(other_, [list(tf.keras.backend.int_shape(o)) for o in other]):
        o.set_shape([max_detections] + s[1:])

    return [boxes, scores, labels, predicate_scores, predicate_labels] + other_


class FilterDetections(tf.keras.layers.Layer):
    """ Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
            self,
            nms=True,
            class_specific_filter=True,
            nms_threshold=0.5,
            score_threshold=0.05,
            max_detections=300,
            parallel_iterations=32,
            **kwargs
    ):
        """ Filters detections using score threshold, NMS and selecting the top-k detections.
        Args
            nms (bool, optional): Flag to enable/disable NMS.
            class_specific_filter (bool, optional): Whether to perform filtering per class,
                or take the best scoring class and filter those.
            nms_threshold (float, optional): Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold (float, optional): Threshold used to pre-filter the boxes with.
            max_detections (int, optional): Maximum number of detections to keep.
            parallel_iterations (int, optional): Number of batch items to process in parallel.
        """
        self.nms = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.parallel_iterations = parallel_iterations
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ Constructs the NMS graph.
        Args
            inputs : List of [boxes, classification, other[0], other[1], ...] tensors.
        """
        boxes = inputs[0]
        classification = inputs[1]
        relationship = inputs[2]
        other = inputs[3:]

        # Wrap nms with our parameters.
        def _filter_detections(args):
            _boxes = args[0]
            _classification = args[1]
            _relationship = args[2]
            _other = args[3:]

            return filter_detections(
                _boxes,
                _classification,
                _relationship,
                _other,
                nms=self.nms,
                class_specific_filter=self.class_specific_filter,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
                nms_threshold=self.nms_threshold,
            )

        # Call filter_detections on each batch.
        outputs = tf.map_fn(
            _filter_detections,
            elems=[boxes, classification, relationship, other],
            dtype=[tf.keras.backend.floatx(), tf.keras.backend.floatx(), 'int32'] + [o.dtype for o in other],
            parallel_iterations=self.parallel_iterations
        )

        return outputs

    def compute_output_shape(self, input_shape):
        """ Computes the output shapes given the input shapes.
        Args
            input_shape : List of input shapes [boxes, classification, other[0], other[1], ...].
        Returns
            List of tuples representing the output shapes:
                [
                    filtered_boxes.shape,
                    filtered_scores.shape,
                    filtered_labels.shape,
                    filtered_other[0].shape,
                    filtered_other[1].shape,
                    ...
                ]
        """
        return [
                   (input_shape[0][0], self.max_detections, 4),
                   (input_shape[1][0], self.max_detections),
                   (input_shape[1][0], self.max_detections),
                   (input_shape[2][0], self.max_detections),
                   (input_shape[2][0], self.max_detections),
               ] + [
                   tuple([input_shape[i][0], self.max_detections] + list(input_shape[i][2:])) for i in
                   range(3, len(input_shape))
               ]

    def compute_mask(self, inputs, mask=None):
        """ This is required in tf.keras when there is more than 1 output.
        """
        return (len(inputs) + 2) * [None]

    def get_config(self):
        """ Gets the configuration of this layer.
        Returns
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
