from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, SpatialDropout2D
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow import keras
from keras.models import load_model
import tensorflow_hub as hub

import numpy as np
from numpy import reshape
import tensorflow as tf
from PIL import Image

def gen_labels():
    train_path = '../Data/train'
    generator_train = ImageDataGenerator(
        rescale = 1.0/255.0,
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        horizontal_flip=True,
        vertical_flip=False
    )

    train_generator = generator_train.flow_from_directory(
        train_path,
        class_mode='categorical',
        target_size=[128, 128],
        color_mode="rgb",
        batch_size=64,
    )

    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())

    return labels

def preprocess(image):
    image = np.array(image.resize((128, 128)))
    image = np.array(image)/255.0
    return image

label_array = ['scab', 'black rot', 'cedar rust', 'healthy', 'healthy', 'healthy', 'powder mildew', 'spot grey leaf', 'common rust', 'healthy', 'leaf blight', 'black rot', 'black measles', 'healthy', 'leaf blight', 'citrus greening', 'bacterial spot', 'healthy', 'bacterial spot', 'healthy', 'early blight', 'healthy', 'late blight', 'healthy', 'healthy', 'powder mildew', 'healthy', 'leaf scorch', 'bacterial spot', 'early blight', 'healthy', 'late blight', 'leaf mold', 'septoria leaf spot', 'two spotted spider mite', 'target spot', 'mosaic virus', 'yellow leaf curl virus']

def model_arc():
    class HubLayer(tf.keras.layers.Layer):
        def __init__(self, handle, **kwargs):
            self.handle = handle
            super(HubLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            self.hub_layer = hub.KerasLayer(self.handle, trainable=False)
            super(HubLayer, self).build(input_shape)

        def call(self, inputs, **kwargs):
            return self.hub_layer(inputs)

        def get_config(self):
            config = super(HubLayer, self).get_config()
            config.update({"handle": self.handle})
            return config

    MODULE_HANDLE = "https://www.kaggle.com/models/google/mobilenet-v2/frameworks/TensorFlow2/variations/035-128-classification/versions/2"
    module = HubLayer(MODULE_HANDLE, input_shape=(128,128,3), trainable=False)

    num_classes = 38
    model = tf.keras.models.Sequential([
        module,
        tf.keras.layers.Dense(1024, activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Replace 'path/to/your/model.h5' with the actual path to your HDF5 model file
    # model_path = '../model/plantmodeltype2.h5'

    # Define a custom object scope to tell TensorFlow about the custom layer
    # custom_objects = {'HubLayer': HubLayer}

    # Load the model using the custom object scope
    # with tf.keras.utils.custom_object_scope(custom_objects):
    #     model = tf.keras.models.load_model(model_path)

    return model
