# PyPi Imports
import tensorflow as tf


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
                                      antialias=False,
                                      name=None)
    return reshaped_tensor
