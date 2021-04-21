"""
Project 4 - CNNs
CS1430 - Computer Vision
Brown University
"""

import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import hyperparameters as hp
from models import SimpleModel, ComplexModel
from preprocess import Datasets
from skimage.transform import resize
from tensorboard_utils import \
        ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver

from skimage.io import imread
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

from create_results_webpage import create_results_webpage
from helpers import get_image_paths
from live import liveApp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--task',
        required=True,
        choices=['1', '3'], # task 1 is simpleModel and task 3 is complexModel 
        help='''Which task of the assignment to run -
        training the SimpleModel (1), or training the ComplexModel(3).''')
    parser.add_argument(
        '--data',
        default='..'+os.sep+'data'+os.sep,
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')
    parser.add_argument(
        '--aug',
        default='1',
        help='''Either 1 for less augmented data and 3 for more augmented data.''')
    parser.add_argument(
        '--live',
        default=None,
        help='''Use the live video application.''')

    return parser.parse_args()



def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, ARGS.task, hp.max_num_weights)
    ]

    # Include confusion logger in callbacks if flag set
    if ARGS.confusion:
        callback_list.append(ConfusionMatrixLogger(logs_path, datasets))

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )


def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )

    categories = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    abbr_categories = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    num_train_per_cat = 100
    train_image_paths, test_image_paths, train_labels, test_labels = \
        get_image_paths('../data/', categories, num_train_per_cat)
    predicted_categories = model.predict(
        x=test_data,
        verbose=1,
    )
    predicted = []
    for j in predicted_categories:
        i = np.argmax(j)
        if (i == 0):
            predicted.append('angry')
        elif (i == 1):
            predicted.append('disgust')
        elif (i == 2):
            predicted.append('fear')
        elif (i == 3):
            predicted.append('happy')
        elif (i == 4):
            predicted.append('sad')
        elif (i == 5):
            predicted.append('surprise')
        elif (i == 6):
            predicted.append('neutral')

    print("predicted", len(predicted))
    print("test labels ", len(test_labels))
    create_results_webpage( train_image_paths, \
                            test_image_paths, \
                            train_labels, \
                            test_labels, \
                            categories, \
                            abbr_categories, \
                            np.array(predicted))
    


def main():
    """ Main function. """

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    # If loading from a checkpoint, the loaded checkpoint's directory
    # will be used for future checkpoints
    if ARGS.load_checkpoint is not None:
        ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

        # Get timestamp and epoch from filename
        regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
        init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
        timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    # If paths provided by program arguments are accurate, then this will
    # ensure they are used. If not, these directories/files will be
    # set relative to the directory of run.py
    if os.path.exists(ARGS.data):
        ARGS.data = os.path.abspath(ARGS.data)

    # Run script from location of run.py
    os.chdir(sys.path[0])

    datasets = Datasets(ARGS.data, ARGS.task, ARGS.aug)
    print("Data set up done")
    if ARGS.task == '1':
        
        model = SimpleModel()
        model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
        checkpoint_path = "checkpoints" + os.sep + \
            "simple_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "simple_model" + \
            os.sep + timestamp + os.sep

        # Print summary of model
        model.summary()
        
    else:
        model = ComplexModel()
        model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
        checkpoint_path = "checkpoints" + os.sep + \
            "complex_model" + os.sep + timestamp + os.sep
        logs_path = "logs" + os.sep + "complex_model" + \
            os.sep + timestamp + os.sep

        # Print summary of model
        model.summary()

    # Load checkpoints
    if ARGS.load_checkpoint is not None:
        model.load_weights(ARGS.load_checkpoint, by_name=False)

    # Make checkpoint directory if needed
    if not ARGS.evaluate and not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])
    
    if ARGS.evaluate:
        test(model, datasets.test_data)

        # TODO: change the image path to be the image of your choice by changing
        # the lime-image flag when calling run.py to investigate
        # i.e. python run.py --evaluate --lime-image test/Bedroom/image_003.jpg
    else:
        train(model, datasets, checkpoint_path, logs_path, init_epoch)
    if ARGS.live:
        

# Make arguments global
ARGS = parse_args()

main()
