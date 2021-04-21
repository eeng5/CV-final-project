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
class LiveApp:
    def __init__(self):
        os.chdir(sys.path[0])
        self.model_path= "checkpoints/simple_model/041321-113618/your.weights.e015-acc0.6121.h5"
        doLive()
    def loadModel(self):
        model = SimpleModel()
        model(tf.keras.Input(shape=(hp.img_size, hp.img_size,3)))
        model.load_weights(self.model_path, by_name=False)
        model.compile(
            optimizer=model.optimizer,
            loss=model.loss_fn,
            metrics=["sparse_categorical_accuracy"],
        )
        return model
    def createPixelArray(self, arr):
        array = np.array(arr, dtype=np.uint8)
        array = cv2.resize(array, (48, 48))
        img = array / 255.
        return img
    def doLive(self):
        model = self.loadModel()
        # load in face detector to get face
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        vs = VideoStream(src=0).start()
        start = time.perf_counter()
        data = []
        time_value = 0
        out = cv2.VideoWriter(
            "liveoutput.avi", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10, (450, 253)
        )
        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=450)
            gray = frame
            face_coord = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
            for coords in face_coord:
                X, Y, w, h = coords
                H, W, _ = frame.shape
                X_1, X_2 = (max(0, X - int(w)), min(X + int(1.3 * w), W))
                Y_1, Y_2 = (max(0, Y - int(0.1 * h)), min(Y + int(1.3 * h), H))
                img_cp = gray[Y_1:Y_1+48, X_1:X_1+48].copy()
                img_mod = createPixelArray(img_cp)
                img_mod = np.expand_dims(img_mod, 0)
                prediction = model.predict(img_mod)
                p = np.argmax(prediction)
                caption = ''
                if (p == 0):
                    caption = 'Angry'
                elif (p == 1):
                    caption = 'Disgust'
                elif (p == 2):
                    caption = 'Fear'
                elif (p == 3):
                    caption = 'Happy'
                elif (p == 4):
                    caption = 'Sad'
                elif (p == 5):
                    caption = 'Surprise'
                elif (p == 6):
                    caption = 'Neutral'
                cv2.rectangle(
                    img=frame,
                    pt1=(X_1, Y_1),
                    pt2=(X_2, Y_2),
                    color=(128, 128, 0),
                    thickness=2,
                )
                cv2.putText(
                    frame,
                    caption,
                    (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (225, 255, 255),
                    2,)
            cv2.imshow("frame", frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        vs.stop()
        out.release()
        cv2.destroyAllWindows()
        