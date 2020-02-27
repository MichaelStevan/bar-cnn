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

####
#### ???
# # invoke modifier if given
#     if modifier:
#         resnet = modifier(resnet)
#### ???
####

# Builtin Imports
import os

# PyPi Library Imports
import tensorflow as tf

# Local Imports
from bar_cnn import custom
from bar_cnn import utils


class RetinaNet:
    """ RetinaNet Model Class in TF 2.0 Keras.

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

    def __init__(self,
                 input_tensors,
                 output_tensors,
                 backbone_layers,
                 pyramid_feature_size=256,
                 regression_feature_size=256,
                 regression_model_head_name="regression_model_head",
                 classification_feature_size=256,
                 classification_prior_probability=0.01,
                 classification_model_head_name="classification_model_head",
                 relationship_feature_size=256,
                 relationship_prior_probability=0.01,
                 relationship_model_head_name="relationship_model_head",
                 num_classes=100,
                 num_predicates=100,
                 num_anchors=None,
                 sub_models=None,
                 nms=True,
                 anchor_params=None,
                 class_specific_filter=True,
                 auto_build=True,
                 model_name='RetinaNet-Box-Attention-Model'):
        """ Constructor method for this class

        Instantiate the class with the given args.
        __init__ is called when ever an object of the class is constructed.

        Args:
            input_tensors (List of Tensors): TODO
            output_tensors (List of Tensors): TODO
            backbone_layers (List of Layers): TODO
            pyramid_feature_size (int, optional): TODO
            regression_feature_size (int, optional): TODO
            regression_model_head_name (str, optional):
            num_classes (int, optional): TODO
            num_predicates (int, optional): TODO
            num_anchors (int, optional): TODO
            sub_models (): TODO
            nms (): TODO
            anchor_params (): TODO
            class_specific_filter (): TODO
            auto_build (bool, optional): Whether or not the model should be built automatically
            model_name (str, optional): TODO
        """

        self.input_tensors = input_tensors
        self.output_tensors = output_tensors
        self.backbone_layers = backbone_layers
        self.pyramid_feature_size = pyramid_feature_size
        self.regression_feature_size = regression_feature_size
        self.regression_model_head_name = regression_model_head_name
        self.classification_feature_size = classification_feature_size
        self.classification_prior_probability = classification_prior_probability
        self.classification_model_head_name = classification_model_head_name
        self.relationship_feature_size = relationship_feature_size
        self.relationship_prior_probability = relationship_prior_probability
        self.relationship_model_head_name = relationship_model_head_name
        self.num_classes = num_classes
        self.num_predicates = num_predicates
        self.num_anchors = num_anchors
        self.nms = nms
        self.anchor_params = anchor_params
        self.class_specific_filter = class_specific_filter
        self.auto_build = auto_build
        self.model_name = model_name

        # To be defined later

        self.model_history = None

        # Instantiate the AnchorParameters class without passing any arguments
        # this will result in the default arguments being applied
        self.anchor_parameters = utils.anchors.AnchorParameters()

        if num_anchors is not None:
            self.num_anchors = num_anchors
        else:
            # Utilize the class method to retrieve the default number of anchors
            self.num_anchors = self.anchor_parameters.num_anchors

        if sub_models is not None:
            self.sub_models = sub_models
        else:
            self.sub_models = self.define_default_sub_models()

        # Whether to run built in functions automatically
        if self.auto_build:
            # Use functions to define more complicated attributes
            self.pyramid_features = self.create_pyramid_features()
            self.pyramid_feature_outputs = [p.output for p in self.pyramid_features]
            self.pyramids = self.build_model_pyramid()

            self.model = self.build_architecture()
            self.loss_fns = self.define_loss_fns()
            self.loss_wts = self.define_loss_wts()
            self.optimizer = self.define_optimizer()
            self.output_path = self.define_output_path()
            self.callbacks = self.define_callbacks()
            # self.class_weights = self.define_class_wts()

            # Print statement yielding information for the user
            print("\n\nMODEL IS NOW BUILT ...\nYOU CAN NOW MANUALLY RUN ...\n\t"
                  "<model_instance>.compile_model() AND <model_instance>.fit_model(tf_dataset)\n\n")

        else:
            self.model = None
            self.loss_fns = None
            self.loss_wts = None
            self.optimizer = None
            self.output_path = None
            self.callbacks = None
            self.class_weights = None

            # Print statement yielding information for the user
            print("\n\nMODEL IS PARTIALLY BUILT ...\n"
                  "PLEASE RUN THE FOLLOWING BUILD STEPS:...\n\t"
                  "1. <model_instance>.build_architecture()\t"
                  "2. <model_instance>.define_loss_fns()\n\t"
                  "3. <model_instance>.define_loss_wts()\n\t"
                  "4. <model_instance>.define_optimizer()\n\t"
                  "5. <model_instance>.define_output_path()\n\t"
                  "6. <model_instance>.define_callbacks()\n\t"
                  "7. <model_instance>.define_class_wts()"
                  "\n\nWHEN YOU ARE FINISHED ...\nYOU WILL BE ABLE TO MANUALLY RUN ...\n\t"
                  "<model_instance>.compile_model() AND <model_instance>.fit_model(tf_dataset)\n\n")

    @staticmethod
    def define_loss_fns():
        """ Defines the loss functions to be used"""
        return {'regression': custom.losses.smooth_l1(),
                'classification': custom.losses.focal(),
                'relationship': custom.losses.focal()
                }

    @staticmethod
    def define_loss_wts():
        """ Defines the loss weightings if they are different"""
        return {
            'regression': 1.0,
            'classification': 1.0,
            'relationship': 1.0
        }

    def define_optimizer(self):
        """ Defines the optimizer based on the passed string"""
        if self.optimizer_name == "ADAM":
            return tf.keras.optimizers.Adam(lr=self.lr, clipnorm=self.clip_norm)

        elif self.optimizer_name == "NADAM":
            print("\n\nNADAM OPTIMIZER NOT YET IMPLEMENTED\n\n... "
                  "defaulting to regular ADAM.\n\n")
            return tf.keras.optimizers.Adam(lr=self.lr, clipnorm=self.clip_norm)

        elif self.optimizer_name == "SGD":
            return tf.keras.optimizers.SGD(lr=self.lr, clipnorm=self.clip_norm)

        else:
            print("\n\n{} OPTIMIZER NOT YET IMPLEMENTED\n\n... " \
                  "defaulting to regular ADAM.\n\n".format(self.optimizer_name))
            return tf.keras.optimizers.Adam(lr=self.lr, clipnorm=self.clip_norm)

    def define_default_sub_models(self):
        """ Create a list of default sub_models used for object detection.

        The default sub_models are...
            1. a regression submodel        (NORMAL RETINANET)
            2. a classification submodel    (NORMAL RETINANET)
            3. relationship submodel.       (BAR-CNN ADDITION)

        Returns:
            A list of tuples, where the first element is the name of the submodel and the second element is the submodel itself.
        """

        regression_tuple = ('regression', self.define_default_regression_model_head())
        classification_tuple = ('classification', self.define_default_classification_model_head())
        relationship_tuple = ('relationship', self.define_default_relationship_model_head())

        return [regression_tuple, classification_tuple, relationship_tuple]

    def define_default_regression_model_head(self, num_values=4):
        """ Creates the default regression sub-model head

        Args:
            num_values (int, optional): TODO

        Returns:
            A tf.keras.models.Model that predicts regression values for each anchor.
        """

        # All new conv layers except the final one in the
        # RetinaNet (classification) subnets are initialized
        # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
        #
        # Define common options to pass into all convolutional layer ops
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'kernel_initializer': tf.random_normal_initializer(mean=0.0, stddev=0.01),
            'bias_initializer': 'zeros'
        }

        # Correct if channels_first was entered - TODO
        if tf.keras.backend.image_data_format() == 'channels_first':
            inputs = tf.keras.layers.Input(shape=(self.pyramid_feature_size, None, None))
        else:
            inputs = tf.keras.layers.Input(shape=(None, None, self.pyramid_feature_size))

        # Assign intermediary variable to pass changes
        x = inputs

        # Iterate through the pyramid convolutions
        for i in range(4):
            x = tf.keras.layers.Conv2D(filters=self.regression_feature_size,
                                       activation='relu',
                                       name='pyramid_regression_{}'.format(i),
                                       **options)(x)

        # Add final convolutional layer to pyramid of regression head
        x = tf.keras.layers.Conv2D(self.num_anchors * num_values,
                                   name='pyramid_regression',
                                   **options)(x)

        # Correct if channels_first was entered - TODO
        if tf.keras.backend.image_data_format() == 'channels_first':
            x = tf.keras.layers.Permute(dims=(2, 3, 1),
                                        name='pyramid_regression_permute')(x)

        # Reshape it to predict 4 values (box corners)
        outputs = tf.keras.layers.Reshape(target_shape=(-1, num_values),
                                          name='pyramid_regression_reshape')(x)

        return tf.keras.models.Model(inputs=inputs,
                                     outputs=outputs,
                                     name=self.regression_model_head_name)

    def define_default_classification_model_head(self):
        """ Creates the default classification sub-model head

        Returns:
            A tf.keras.models.Model that predicts classes for each anchor.
        """

        # Define common options to pass into all convolutional layer ops
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'kernel_initializer': tf.random_normal_initializer(mean=0.0, stddev=0.01),
        }

        # Fix this with the channels stuff - TODO
        if tf.keras.backend.image_data_format() == 'channels_first':
            inputs = tf.keras.layers.Input(shape=(self.pyramid_feature_size, None, None))
        else:
            inputs = tf.keras.layers.Input(shape=(None, None, self.pyramid_feature_size))

        x = inputs
        for i in range(4):
            x = tf.keras.layers.Conv2D(filters=self.classification_feature_size,
                                       activation='relu',
                                       name='pyramid_classification_{}'.format(i),
                                       bias_initializer='zeros',
                                       **options)(x)

        x = tf.keras.layers.Conv2D(filters=self.num_classes * self.num_anchors,
                                   bias_initializer=custom.initializers.PriorProbability(
                                       probability=self.classification_prior_probability),
                                   name='pyramid_classification',
                                   **options)(x)

        # Correct for channels first - TODO
        if tf.keras.backend.image_data_format() == 'channels_first':
            x = tf.keras.layers.Permute(dims=(2, 3, 1),
                                        name='pyramid_classification_permute')(x)
        # reshape output and apply sigmoid
        x = tf.keras.layers.Reshape(target_shape=(-1, self.num_classes),
                                    name='pyramid_classification_reshape')(x)
        outputs = tf.keras.layers.Activation(activation='sigmoid',
                                             name='pyramid_classification_sigmoid')(x)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs,
                                     name=self.classification_model_head_name)

    def define_default_relationship_model_head(self):
        """ Creates the default relationship sub-model head.

        Returns:
            A tf.keras.models.Model that predicts predicates for each anchor.
        """

        # Define common options to pass into all convolutional layer ops
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'kernel_initializer': tf.random_normal_initializer(mean=0.0, stddev=0.01),
        }

        # Fix this with the channels stuff - TODO
        if tf.keras.backend.image_data_format() == 'channels_first':
            inputs = tf.keras.layers.Input(shape=(self.pyramid_feature_size, None, None))
        else:
            inputs = tf.keras.layers.Input(shape=(None, None, self.pyramid_feature_size))

        x = inputs
        for i in range(4):
            x = tf.keras.layers.Conv2D(filters=self.relationship_feature_size,
                                       activation='relu',
                                       name='pyramid_relationship_{}'.format(i),
                                       bias_initializer='zeros',
                                       **options)(x)

        x = tf.keras.layers.Conv2D(filters=self.num_predicates*self.num_anchors,
                                   bias_initializer=custom.initializers.PriorProbability(
                                       probability=self.relationship_prior_probability),
                                   name='pyramid_relationship',
                                   **options)(x)

        # Correct for channels first - TODO
        if tf.keras.backend.image_data_format() == 'channels_first':
            x = tf.keras.layers.Permute(dims=(2, 3, 1),
                                        name='pyramid_relationship_permute')(x)

        # reshape output and apply sigmoid
        x = tf.keras.layers.Reshape(target_shape=(-1, self.num_classes),
                                    name='pyramid_relationship_reshape')(x)
        outputs = tf.keras.layers.Activation(activation='sigmoid',
                                             name='pyramid_relationship_sigmoid')(x)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs,
                                     name=self.relationship_model_head_name)

    def define_output_path(self):
        """ Builds path to output directory"""
        full_output_dir_path = os.path.join(self.base_output_path, self.model_name)

        if not os.path.isdir(full_output_dir_path):
            print("\n\nOutput directory does not exist ...\n"
                  "Creating directory\t{}\t...\n\n".format(full_output_dir_path))
            os.makedirs(full_output_dir_path, exist_ok=True)
        return full_output_dir_path

    def define_callbacks(self):
        """ Builds complete callback list from required & additional callbacks"""
        # A list containing a single callback
        base_callback = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.output_path,
                                      "weights.{epoch:02d}-{loss:.2f}.hdf5"),
                monitor='loss',
                save_weights_only=True,
                verbose=1)
        ]
        # Either an empty list (if no additional callbacks) or a list of additional callbacks
        if self.additional_callbacks is not None:
            return base_callback + self.additional_callbacks
        else:
            return base_callback

    def define_class_wts(self):
        """ Defines the class weightings if required"""
        raise NotImplementedError

    def load_weights(self, abs_path_to_weights):
        """ Loads the weight file passed into the current model

        Args:
            abs_path_to_weights (str): TODO
        """
        self.model.load_weights(abs_path_to_weights, by_name=True)

    def compile_model(self):
        """ Completes model compilation process using specified parameters """

        print("[INFO] compiling model ...\n")
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_fns,
                           loss_weights=self.loss_wts,
                           metrics=self.metrics)
        print("\n[INFO] compiling model complete... \n\n")

    def fit_model(self, TF_DS):
        """ Launches model training process using specified parameters

        Args:
            TF_DS (Tensorflow Dataset Object): TODO
        """
        # #################### SOMETHING LIKE THIS #################### #
        # print("[INFO] starting model training ...\n")
        # self.history = self.model.fit(x=TF_DS,
        #                               batch_size=self.batch_size,
        #                               epochs=self.epochs,
        #                               callbacks=self.callbacks)
        # print("\n[INFO] finishing model training ...\n\n")
        # #################### ################### #################### #
        raise NotImplementedError

    def create_pyramid_features(self):
        """ Creates the FPN layers on top of the backbone features.

        Returns:
            A list of feature levels [P3, P4, P5, P6, P7].
        """

        # Get feature stages from the passed backbone layers variable
        C3, C4, C5 = self.backbone_layers

        # upsample C5 to get P5 from the FPN paper
        P5 = tf.keras.layers.Conv2D(filters=self.pyramid_feature_size, kernel_size=1,
                                    strides=1, padding='same', name='C5_reduced')(C5)
        P5_upsampled = custom.layers.UpsampleLike(name='P5_upsampled')([P5, C4])
        P5 = tf.keras.layers.Conv2D(filters=self.pyramid_feature_size, kernel_size=3,
                                    strides=1, padding='same', name='P5')(P5)

        # add P5 element-wise to C4
        P4 = tf.keras.layers.Conv2D(filters=self.pyramid_feature_size, kernel_size=1,
                                    strides=1, padding='same', name='C4_reduced')(C4)
        P4 = tf.keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
        P4_upsampled = custom.layers.UpsampleLike(name='P4_upsampled')([P4, C3])
        P4 = tf.keras.layers.Conv2D(filters=self.pyramid_feature_size, kernel_size=3,
                                    strides=1, padding='same', name='P4')(P4)

        # add P4 element-wise to C3
        P3 = tf.keras.layers.Conv2D(filters=self.pyramid_feature_size, kernel_size=1,
                                    strides=1, padding='same', name='C3_reduced')(C3)
        P3 = tf.keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
        P3 = tf.keras.layers.Conv2D(filters=self.pyramid_feature_size, kernel_size=3,
                                    strides=1, padding='same', name='P3')(P3)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        P6 = tf.keras.layers.Conv2D(filters=self.pyramid_feature_size, kernel_size=3,
                                    strides=2, padding='same', name='P6')(C5)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        P7 = tf.keras.layers.Activation('relu', name='C6_relu')(P6)
        P7 = tf.keras.layers.Conv2D(filters=self.pyramid_feature_size, kernel_size=3,
                                    strides=2, padding='same', name='P7')(P7)

        return [P3, P4, P5, P6, P7]

    def build_model_pyramid(self):
        """

        Returns: List of pyramid ... TODO

        """

        # COULD BE DONE WITH A SINGLE LINE LIST COMP. JUST NOT SURE ABOUT READABILITY
        #
        # [tf.keras.layers.Concatenate(axis=1, name=model_name)([model(f) for f in self.pyramid_features]) \
        #     for model_name, model in self.sub_models]

        model_pyramid = []
        for model_name, model in self.sub_models:
            pyramid_feature_layer_outputs = [model(f) for f in self.pyramid_features]
            model_pyramid.append(tf.keras.layers.Concatenate(
                axis=1, name=model_name)(pyramid_feature_layer_outputs))

        return model_pyramid

    def build_anchors(self):
        """ Builds anchors for the shape of the features from FPN.

        Returns:
            A tensor containing the anchors for the FPN features.
                The shape is: (batch_size, num_anchors, 4)
        """

        anchors = [
            custom.layers.GenerateAnchors(size=self.anchor_parameters.sizes[i],
                                          stride=self.anchor_parameters.strides[i],
                                          ratios=self.anchor_parameters.ratios,
                                          scales=self.anchor_parameters.scales,
                                          name='anchors_{}'.format(i))(f)
            for i, f in enumerate(self.pyramid_feature_outputs)
        ]

        return tf.keras.layers.Concatenate(axis=1, name='anchors')(anchors)

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
