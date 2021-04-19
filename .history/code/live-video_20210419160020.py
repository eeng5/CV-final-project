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
 
 
# mp4, get all images frame by frame
def get_frames():
    vidcap = cv2.VideoCapture('emotions_test_vid.mp4')
    fps = vidcap.get(cv2.CV_CAP_PROP_FPS)
    frames = []
    success, image = vidcap.read()
    count = 0
    while success:
        image = resize(image, (hp.img_size, hp.img_size, 3))
        path = os.sep + 'video_imgs' + os.sep + 'frame' + str(count) + '.jpg'
        cv2.imwrite(path, image)    #save frame as JPEG file
        frames.append(image)
        success, image = vidcap.read()
        ##print('Read a new frame: ', success)
        count += 1
    return frames, fps
 
def main():
    # Get frames
    # Predict each frame's classification
    # Display the results
 
    # Creates frames from the video
    frames, fps = get_frames()
    
    # The path and file of the weights
    weights_str = '/checkpoints/simple_model/041521-215802/your.weights.e005-acc0.2494.h5'
 
    # The path to where the frames are stored
    #data_str = os.sep + 'video_imgs' + os.sep
 
    # Run script from location of run.py
    os.chdir(sys.path[0])
 
    # Initializes a Datasets
    #datasets = Datasets(data_str, '1')
 
    # Initializes a model
    model = SimpleModel()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    model.head.load_weights(weights_str, by_name=False)
    
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])
 
    #print('Testing...')
 
    sorted_imgs = []
    #(237, 230, 211)
    frames_arr = np.array(frames)
    predictions = np.argmax(model.predict(frames_arr))
    for i in range(predictions.shape[0]):
        p = predictions[i]
        f = frames_arr[i]
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
        c_font = ImageFont.truetype('playfair/playfair-font.ttf', 20)
        new_img = ImageDraw.Draw(f)
        new_img.text((15,15), caption, (24, 23, 21), font=c_font)
        sorted_imgs.append(new_img)
 
    # Creates a video from the classified frames
    out = cv2.VideoWriter('test_result_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (hp.img_size, hp.img_size))
    
    for img in sorted_imgs:
        out.write(img)
    out.release()
 
    ##model(tf.keras.Imput(shape=(hp.img)))
 
print(1)
main()
#print('Finish!!')
