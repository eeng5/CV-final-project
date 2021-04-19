
import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf

import hyperparameters as hp
from datasetup import createSimpleData, createComplexData

class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path, task, aug):

        self.data_path = data_path
        self.task = task
        self.aug = aug
        if self.aug == '1':
            createSimpleData(data_path)
        else:
            createComplexData()
        # Dictionaries for (label index) <--> (class name)
        self.idx_to_class = {}
        self.class_to_idx = {}

        # For storing list of classes
        self.classes = [""] * hp.num_classes

        # Setup data generators
        self.train_data = self.get_data(
            os.path.join(self.data_path, "train/"), False)
        self.test_data = self.get_data(
            os.path.join(self.data_path, "test/"), False)

    

    def preprocess_fn(self, img):
        """ Preprocess function for ImageDataGenerator. """
        img = img / 255.
        return img

    def get_data(self, path, shuffle):
        """ Returns an image data generator which can be iterated
        through for images and corresponding class labels.

        Arguments:
            path - Filepath of the data being imported, such as
                   "../data/train" or "../data/test"
            shuffle - Boolean value indicating whether the data should
                      be randomly shuffled.

        Returns:
            An iterable image-batch generator
        """

        
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=self.preprocess_fn)

        # VGG must take images of size 224x224
        img_size = hp.img_size

        classes_for_flow = None

        # Make sure all data generators are aligned in label indices
        if bool(self.idx_to_class):
            classes_for_flow = self.classes

        # Form image data generator from directory structure
        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(img_size, img_size),
            class_mode='sparse',
            batch_size=hp.batch_size,
            shuffle=shuffle,
            classes=classes_for_flow)

        # Setup the dictionaries if not already done
        if not bool(self.idx_to_class):
            unordered_classes = []
            for dir_name in os.listdir(path):
                if os.path.isdir(os.path.join(path, dir_name)):
                    unordered_classes.append(dir_name)

            for img_class in unordered_classes:
                self.idx_to_class[data_gen.class_indices[img_class]] = img_class
                self.class_to_idx[img_class] = int(data_gen.class_indices[img_class])
                self.classes[int(data_gen.class_indices[img_class])] = img_class

        return data_gen