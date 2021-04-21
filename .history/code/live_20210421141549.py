import os
import cv2
import sys

import numpy as np
from models import SimpleModel
from preprocess import Datasets

import hyperparameters as hp
import tensorflow as tf

from skimage.transform import resize
from PIL import Image, ImageFont, ImageDraw


from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import VideoStream
import fastai
import fastai.vision
import imutils
import argparse
import time
import dlib
""" This file is a live video emotion detection application. To run simply activate the virtual environment in the code dirrectory via:
$ source cs14_30/bin/activate
Then run teh below command in the virtual environment:
$ python3 live.py
"""
Def
