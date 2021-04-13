"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, AveragePooling2D
import hyperparameters as hp


class SimpleModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(SimpleModel, self).__init__()

        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp.learning_rate, momentum=hp.momentum)

       ## Conv2D(filters, kernel_size, strides, padding, activation, name)
       ## filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
       ## kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
       ## strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
        self.architecture = [
            Conv2D(64, 3, 1, padding="same", activation="relu"),
            BatchNormalization(),
            Dropout(0.20),
            MaxPool2D(2),
            Conv2D(124, 5, 1, padding="same", activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            MaxPool2D(2),
            Conv2D(512, 3, 1, padding="same", activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            MaxPool2D(2),
            Conv2D(512, 3, 1, padding="same", activation="relu"),

            Flatten(), 
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            Dense(7,  activation='relu'),
            BatchNormalization(),
            Dropout(0.2)
        ]
        
    # above model used softmax as their loss function 


    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """
        
        #Select a loss function for your network (see the documentation
        #       for tf.keras.losses)
        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions, from_logits=False)


class ComplexModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(ComplexModel, self).__init__()

        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp.learning_rate, momentum=hp.momentum)
    
       ## Conv2D(filters, kernel_size, strides, padding, activation, name)
       ## filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
       ## kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
       ## strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
        self.architecture = [
            Conv2D(64, 5, 1, padding="same", activation="relu"),
            MaxPool2D(pool_size=(5,5), strides=(2,2)),

            Conv2D(64, 5, 1, padding="same", activation="relu"),
            Conv2D(64, 5, 1, padding="same", activation="relu"),
            AveragePooling2D(pool_size=(3,3), strides=(2,2)),

            Conv2D(128, 3, 1, padding="same", activation="relu"),
            Conv2D(128, 3, 1, padding="same", activation="relu"),
            AveragePooling2D(pool_size=(3,3), strides=(2,2)),

            Flatten(), 
            Dense(1024, activation="relu"),
            Dropout(0.2), 
            Dense(1024, activation="relu"),
            Dropout(0.2),
            Dense(1024, activation="relu"),
            Dropout(0.2),
            Dense(7, activation="softmax")  
        ]
        


    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """
        
        #Select a loss function for your network (see the documentation
        #       for tf.keras.losses)
        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions, from_logits=False)

