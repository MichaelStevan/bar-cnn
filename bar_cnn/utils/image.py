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
from tensorflow.keras.applications.imagenet_utils import preprocess_input as preprocess_imagenet_input


def preprocess_input(img, method="imagenet"):
    """ Preprocess an image by subtracting the ImageNet mean.

    Args:
        img (np.array): Shape (None, None, 3) or (3, None, None).
        method (optional): What preprocessing method to apply

    Returns:
        The input with the ImageNet mean subtracted.
    """
    if method == "imagenet":
        return preprocess_imagenet_input(img)
    else:
        raise NotImplementedError("Unfortunately only imagenet pre-processing has been implemented as-of-now")
