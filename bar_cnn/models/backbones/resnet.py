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

---

Description:
    ResNet Model for TF 2.xx Augmented for BAR-CNN

NOTE: We overwrite the model that can be found here:
    https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py

Reference:
    - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
    - [Detecting Visual Relationships Using Box Attention](https://arxiv.org/pdf/1807.02136.pdf)
    - [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf)
"""

# PyPi Library Imports
import tensorflow as tf

# Local Imports
from bar_cnn import custom
from bar_cnn import utils


class ResNetBoxAttnBackbone:
    """ ResNet Model Class in TF 2.0 Keras Modified for Box Attention

    Reference papers
    - [Deep Residual Learning for Image Recognition]
        - https://arxiv.org/abs/1512.03385 (CVPR 2016 Best Paper Award)
    - [Identity Mappings in Deep Residual Networks]
        - https://arxiv.org/abs/1603.05027 (ECCV 2016)
    - [Aggregated Residual Transformations for Deep Neural Networks]
        - https://arxiv.org/abs/1611.05431 (CVPR 2017)

    Reference implementations
    - [TensorNets]
        - github.com/taehoonlee/tensornets/blob/master/tensornets/resnets.py
    - [Caffe ResNet]
        - github.com/KaimingHe/deep-residual-networks/tree/master/prototxt
    - [Torch ResNetV2]
        - github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua
    - [Torch ResNeXt]
        - github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua
    """

    def __init__(self, image_size=160,
                 regular_kernel_initializer='he_normal',
                 attention_kernel_initializer='zeros',
                 activation_function='relu',
                 last_pool="avg",
                 utilize_bias=False,
                 padding_style="same",
                 freeze_batch_norm=True,
                 model_name="resnet-bar-cnn",
                 auto_model_build=True):
        """ Constructor method for this class

        Instantiate the class with the given args.
        __init__ is called when ever an object of the class is constructed.


        Args:
            regular_kernel_initializer (str, optional): TODO
            attention_kernel_initializer (str, optional): TODO
            activation_function (str, optional): TODO
            last_pool (str, optional): TODO
            utilize_bias (bool, optional): TODO
            padding_style (str, optional): TODO
            freeze_batch_norm (bool, optional): TODO
            model_name (str, optional): TODO
            auto_model_build (bool, optional): TODO
        """

        self.image_shape = (image_size, image_size, 3)
        self.bn_axis = self.get_bn_axis()
        self.reg_kernel_init = regular_kernel_initializer
        self.attn_kernel_init = attention_kernel_initializer
        self.activation_fn = activation_function
        self.last_pool_style = last_pool
        self.use_bias = utilize_bias
        self.pad_style = padding_style
        self.freeze_bn = freeze_batch_norm
        self.model_name = model_name
        self.auto_build = auto_model_build

        # Initialize stage output array
        # to be populated when using method build_architecture
        self.stage_outputs = []

        # Whether to run built in functions automatically
        if self.auto_build:
            # Use functions to define more complicated attributes
            self.model = self.build_architecture()

        else:
            self.model = None

            # Print statement yielding information for the user
            print("\n\nMODEL IS PARTIALLY DEFINED ...\n"
                  "PLEASE RUN THE FOLLOWING BUILD STEPS:...\n\t"
                  "1. <model_instance>.build_architecture()")

    def load_weights(self, abs_path_to_weights):
        """ Loads the weight file passed into the current model

        Args:
            abs_path_to_weights (str): TODO
        """
        self.model.load_weights(abs_path_to_weights, by_name=True)

    @staticmethod
    def get_bn_axis():
        """ Defines the batch normalization axis"""
        if tf.keras.backend.image_data_format() == 'channels_last':
            return 3
        else:
            return 1

    @staticmethod
    def get_layer_name(prefix, stage, block):
        """ Get layer name built from prefix, stage, and block number

        Args:
            prefix (string): TODO
            stage (string): TODO
            block (string): TODO

        Returns:
            Layer name as a string
        """
        return "{}{}{}_branch".format(str(prefix), str(stage), str(block))

    def identity_block(self,
                       input_tensor,
                       kernel_size,
                       filters,
                       stage,
                       block,
                       attn_map=None):
        """The identity block is the block that has no convolutional layer at shortcut.

        Args:
            input_tensor (Tensor): TODO
            kernel_size (int): the kernel size of middle conv layer at main path
            filters (list of ints): the filters of 3 conv layer at main path
            stage (int): current stage label, used for generating layer names
            block (str): 'a','b'..., current block label, used for generating layer names
            attn_map (TODO, optional): TODO (defaults to None)

        Returns:
            Output tensor for the block.
        """

        # map filter size to filers1, filters2, and filters3
        filters1, filters2, filters3 = filters

        # Names take the form "<prefix><stage><block>_branch"
        conv_name_base = self.get_layer_name(prefix="res", stage=stage, block=block)
        bn_name_base = self.get_layer_name(prefix="bn", stage=stage, block=block)
        attn_map_name_base = self.get_layer_name(prefix="attn", stage=stage, block=block)

        x = tf.keras.layers.Conv2D(filters=filters1, kernel_size=(1, 1), use_bias=self.use_bias,
                                   kernel_initializer=self.reg_kernel_init,
                                   name=conv_name_base + '2a')(input_tensor)

        x = custom.layers.BatchNormalization(freeze=self.freeze_bn, axis=self.bn_axis,
                                             name=bn_name_base + '2a')(x)
        x = tf.keras.layers.Activation(self.activation_fn)(x)

        # This is the convolutional layer that we want to modify.
        x = tf.keras.layers.Conv2D(filters=filters2, kernel_size=kernel_size,
                                   padding=self.pad_style, use_bias=self.use_bias,
                                   kernel_initializer=self.reg_kernel_init,
                                   name=conv_name_base + '2b')(x)

        # Get the output shape of the layer
        attn_map_shaped = tf.keras.layers.Lambda(function=utils.tf.resize_like,
                                                 arguments={'ref_tensor': x})(attn_map)

        # Get the number of filters required
        num_attn_filters = x.shape[3]

        x = custom.layers.BatchNormalization(freeze=self.freeze_bn, axis=self.bn_axis,
                                             name=bn_name_base + '2b')(x)

        # Do a 3x3 convolution on the attention map
        attention_layer = tf.keras.layers.Conv2D(filters=num_attn_filters, kernel_size=3,
                                                 padding=self.pad_style, use_bias=self.use_bias,
                                                 kernel_initializer=self.attn_kernel_init,
                                                 name=attn_map_name_base + '2b')(attn_map_shaped)
        x = tf.keras.layers.add(inputs=[x, attention_layer])
        x = tf.keras.layers.Activation(self.activation_fn)(x)

        x = tf.keras.layers.Conv2D(filters=filters3,
                                   kernel_size=(1, 1),
                                   use_bias=self.use_bias,
                                   kernel_initializer=self.reg_kernel_init,
                                   name=conv_name_base + '2c')(x)

        x = custom.layers.BatchNormalization(freeze=self.freeze_bn,
                                             axis=self.bn_axis,
                                             name=bn_name_base + '2c')(x)
        x = tf.keras.layers.add(inputs=[x, input_tensor])
        x = tf.keras.layers.Activation(self.activation_fn)(x)
        return x

    def conv_block(self,
                   input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   strides=(2, 2),
                   attn_map=None):
        """A block that has a conv layer at shortcut.

        Args:
            input_tensor: (TODO) input tensor
            kernel_size: (TODO) default 3, the kernel size of middle conv layer at main path
            filters: (TODO) list of integers, the filters of 3 conv layer at main path
            stage: (TODO) integer, current stage label, used for generating layer names
            block: (TODO) 'a','b'..., current block label, used for generating layer names
            strides: (TODO, optional) Strides for the first conv layer in the block. (defaults to (2,2))
            attn_map: (TODO, optional) TODO (defaults to None)

        Returns:
            Output tensor for the block.

        Note that from stage 3, the first conv layer at main path is with strides=(2, 2)
        Note that the shortcut should have strides=(2, 2)
        """
        filters1, filters2, filters3 = filters

        # Names take the form "<prefix><stage><block>_branch"
        conv_name_base = self.get_layer_name(prefix="res", stage=stage, block=block)
        bn_name_base = self.get_layer_name(prefix="bn", stage=stage, block=block)
        attn_map_name_base = self.get_layer_name(prefix="attn", stage=stage, block=block)

        x = tf.keras.layers.Conv2D(filters=filters1, kernel_size=(1, 1),
                                   strides=strides, use_bias=self.use_bias,
                                   kernel_initializer=self.reg_kernel_init,
                                   name=conv_name_base + '2a')(input_tensor)

        x = custom.layers.BatchNormalization(freeze=self.freeze_bn, axis=self.bn_axis,
                                             name=bn_name_base + '2a')(x)
        x = tf.keras.layers.Activation(self.activation_fn)(x)

        # This is the convolutional layer that we want to modify.
        x = tf.keras.layers.Conv2D(filters=filters2, kernel_size=kernel_size,
                                   padding=self.pad_style, use_bias=self.use_bias,
                                   kernel_initializer=self.reg_kernel_init,
                                   name=conv_name_base + '2b')(x)

        # Get the output shape of the layer
        attn_map_shaped = tf.keras.layers.Lambda(function=utils.tf.resize_like,
                                                 arguments={'ref_tensor': x})(attn_map)

        # Get the number of filters required
        num_attn_filters = x.shape[3]

        x = custom.layers.BatchNormalization(freeze=self.freeze_bn, axis=self.bn_axis,
                                             name=bn_name_base + '2b')(x)

        # Do a 3x3 convolution on the attention map
        attention_layer = tf.keras.layers.Conv2D(filters=num_attn_filters, kernel_size=3,
                                                 padding=self.pad_style, use_bias=self.use_bias,
                                                 kernel_initializer=self.attn_kernel_init,
                                                 name=attn_map_name_base + '2b')(attn_map_shaped)

        x = tf.keras.layers.add(inputs=[x, attention_layer])
        x = tf.keras.layers.Activation(self.activation_fn)(x)
        x = tf.keras.layers.Conv2D(filters=filters3, kernel_size=(1, 1), use_bias=self.use_bias,
                                   kernel_initializer=self.reg_kernel_init,
                                   name=conv_name_base + '2c')(x)

        x = custom.layers.BatchNormalization(freeze=self.freeze_bn, axis=self.bn_axis,
                                             name=bn_name_base + '2c')(x)

        shortcut = tf.keras.layers.Conv2D(filters=filters3, kernel_size=(1, 1),
                                          strides=strides, use_bias=self.use_bias,
                                          kernel_initializer=self.reg_kernel_init,
                                          name=conv_name_base + '1')(input_tensor)

        shortcut = custom.layers.BatchNormalization(freeze=self.freeze_bn, axis=self.bn_axis,
                                                    name=bn_name_base + '1')(shortcut)

        x = tf.keras.layers.add(inputs=[x, shortcut])
        x = tf.keras.layers.Activation(self.activation_fn)(x)
        return x

    def build_architecture(self, version="resnet50"):
        """Instantiates the ResNet50 architecture

        Args:
            version (str, optional): which style of resnet are using

        Returns:
            A tf.keras model object.

        Raises:
            NotImplementedError: If passed version has not yet been implemented
        """

        # Version checking - currently only ResNet50 is supported
        if version == "resnet50":
            pass
        else:
            raise NotImplementedError

        # Define inputs
        img_input = tf.keras.layers.Input(shape=self.image_shape)
        attn_map_input = tf.keras.layers.Input(shape=self.image_shape)

        # ################################################################## #
        #                     BUILD STAGE 1 (INPUT) BLOCK                    #
        # ################################################################## #
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                                   padding=self.pad_style, use_bias=self.use_bias,
                                   kernel_initializer=self.reg_kernel_init,
                                   name='conv1')(img_input)

        x = custom.layers.BatchNormalization(freeze=self.freeze_bn, axis=self.bn_axis,
                                             name='bn_conv1')(x)
        x = tf.keras.layers.Activation(self.activation_fn)(x)
        x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        # ################################################################## #
        # ################################################################## #
        # ################################################################## #

        # ################################################################## #
        #                        BUILD STAGE 2 BLOCK                         #
        # ################################################################## #
        x = self.conv_block(input_tensor=x, kernel_size=3, filters=[64, 64, 256],
                            stage=2, block='a', strides=(1, 1), attn_map=attn_map_input)
        x = self.identity_block(input_tensor=x, kernel_size=3, filters=[64, 64, 256],
                                stage=2, block='b', attn_map=attn_map_input)
        x = self.identity_block(input_tensor=x, kernel_size=3, filters=[64, 64, 256],
                                stage=2, block='c', attn_map=attn_map_input)
        self.stage_outputs.append(x)
        # ################################################################## #
        # ################################################################## #
        # ################################################################## #

        # ################################################################## #
        #                        BUILD STAGE 3 BLOCK                         #
        # ################################################################## #
        x = self.conv_block(input_tensor=x, kernel_size=3, filters=[128, 128, 512],
                            stage=3, block='a', attn_map=attn_map_input)
        x = self.identity_block(input_tensor=x, kernel_size=3, filters=[128, 128, 512],
                                stage=3, block='b', attn_map=attn_map_input)
        x = self.identity_block(input_tensor=x, kernel_size=3, filters=[128, 128, 512],
                                stage=3, block='c', attn_map=attn_map_input)
        x = self.identity_block(input_tensor=x, kernel_size=3, filters=[128, 128, 512],
                                stage=3, block='d', attn_map=attn_map_input)
        self.stage_outputs.append(x)
        # ################################################################## #
        # ################################################################## #
        # ################################################################## #

        # ################################################################## #
        #                        BUILD STAGE 4 BLOCK                         #
        # ################################################################## #
        x = self.conv_block(input_tensor=x, kernel_size=3, filters=[256, 256, 1024],
                            stage=4, block='a', attn_map=attn_map_input)
        x = self.identity_block(input_tensor=x, kernel_size=3, filters=[256, 256, 1024],
                                stage=4, block='b', attn_map=attn_map_input)
        x = self.identity_block(input_tensor=x, kernel_size=3, filters=[256, 256, 1024],
                                stage=4, block='c', attn_map=attn_map_input)
        x = self.identity_block(input_tensor=x, kernel_size=3, filters=[256, 256, 1024],
                                stage=4, block='d', attn_map=attn_map_input)
        x = self.identity_block(input_tensor=x, kernel_size=3, filters=[256, 256, 1024],
                                stage=4, block='e', attn_map=attn_map_input)
        x = self.identity_block(input_tensor=x, kernel_size=3, filters=[256, 256, 1024],
                                stage=4, block='f', attn_map=attn_map_input)
        self.stage_outputs.append(x)
        # ################################################################## #
        # ################################################################## #
        # ################################################################## #

        # ################################################################## #
        #                        BUILD STAGE 5 BLOCK                         #
        # ################################################################## #
        x = self.conv_block(input_tensor=x, kernel_size=3, filters=[512, 512, 2048],
                            stage=5, block='a', attn_map=attn_map_input)
        x = self.identity_block(input_tensor=x, kernel_size=3, filters=[512, 512, 2048],
                                stage=5, block='b', attn_map=attn_map_input)
        x = self.identity_block(input_tensor=x, kernel_size=3, filters=[512, 512, 2048],
                                stage=5, block='c', attn_map=attn_map_input)

        # Determine type of pooling to use for final pooling layer
        # and add this layer to the network
        if self.last_pool_style == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif self.last_pool_style == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)
        else:
            raise NotImplementedError
        self.stage_outputs.append(x)
        # ################################################################## #
        # ################################################################## #
        # ################################################################## #

        # ################################################################## #
        #                           MODEL CREATION                           #
        # ################################################################## #
        model = tf.keras.models.Model(inputs=[img_input, attn_map_input],
                                      outputs=self.stage_outputs,
                                      name=self.model_name)
        # ################################################################## #
        # ################################################################## #
        # ################################################################## #

        return model
