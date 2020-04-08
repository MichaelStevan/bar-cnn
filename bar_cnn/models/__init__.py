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

# Local Imports
from bar_cnn.models import backbones


def load_model(filepath, backbone_name='resnet'):
    """ Loads model using the correct custom objects (backbone, etc.)

    Args:
        filepath (str): one of the following:
            - string, path to the saved model, or
            - filepath from which to load the model
        backbone_name (str, optional): Backbone with which the model was trained. (defaults to 'resnet50')

    Returns:
        A tf.keras.models.Model object.

    Raises
        ValueError: In case of an invalid savefile or path

    """
    # TODO
    if file_path:
        pass

    # TODO
    if backbone_name == "resnet":
        backbone = backbones.resnet
    else:
        raise NotImplementedError("No Other Backbones Have Been Implemented\n\n")

    raise NotImplementedError("This whole function is not yet implemented")
    # SOMETHING LIKE THIS
    # return tf.keras.models.load_model(filepath, custom_objects=backbone.custom_objects)
