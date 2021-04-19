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
from fastai.vision import *
import imutils
import argparse
import time
import dlib

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
vs = VideoStream(src=0).start()
start = time.perf_counter() 
data = []
time_value = 0
out = cv2.VideoWriter(path + "liveoutput.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 10, (450,253))
while True:    
    frame = vs.read()    
    frame = imutils.resize(frame, width=450)     
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
    face_coord = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    for coords in face_coord:        
        X, Y, w, h = coords        
        H, W, _ = frame.shape        
        X_1, X_2 = (max(0, X - int(w * 0.3)), min(X + int(1.3 * w), W))        
        Y_1, Y_2 = (max(0, Y - int(0.3 * h)), min(Y + int(1.3 * h), H))        
        img_cp = gray[Y_1:Y_2, X_1:X_2].copy()        
        prediction, idx, probability = learn.predict(Image(pil2tensor(img_cp, np.float32).div_(225)))
        cv2.rectangle(                
            img=frame,                
            pt1=(X_1, Y_1),                
            pt2=(X_2, Y_2),                
            color=(128, 128, 0),                
            thickness=2,            
        )        
       cv2.putText(frame, str(prediction), (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 255, 255), 2)
    cv2.imshow("frame", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
vs.stop()
out.release()
cv2.destroyAllWindows()