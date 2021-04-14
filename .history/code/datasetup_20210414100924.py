from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import cv2
import os
import glob
from pathlib import Path


def cleanTestDirs():
    for f in Path('/Users/Natalie/Desktop/cs1430/CV-final-project/data/test').glob('*.jpg'):
        try:
            f.unlink()
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))
def cleanTrainDirs():
    for f in Path('/Users/Natalie/Desktop/cs1430/CV-final-project/data/train').glob('*.jpg'):
        try:
            f.unlink()
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))
def cleanAll():
    cleanTestDirs()
    cleanTrainDirs()
def createPixelArray(arr):
  arr = list(map(int, arr.split()))
  array = np.array(arr, dtype=np.uint8)
  array = array.reshape((48, 48))
  return array
def equalize_hist(img):
  img = cv2.equalizeHist(img)
  return img
def showImages(imgs):
  _, axs = plt.subplots(1, len(imgs), figsize=(20, 20))
  axs = axs.flatten()
  for img, ax in zip(imgs, axs):
    ax.imshow(img,cmap=plt.get_cmap('gray'))
  plt.show()
def augmentIMG(img, task):
    imgs = [img]
    img1 = equalize_hist(img)
    imgs.append(img1)
    if(task == 3):
        img2 = cv2.bilateralFilter(img1, d=9, sigmaColor=75, sigmaSpace=75)
        imgs.append(img2)
    img6 = cv2.flip(img, 1) # flip horizontally
    imgs.append(img6)
    return imgs
def saveIMG(arr, num, folderLoc):
    im = Image.fromarray(arr)
    filename = folderLoc + "image_"+ num+".jpg"
    im.save(filename)
def createTrain(emotion_dict, task):
    df = pd.read_csv('/Users/Natalie/Desktop/cs1430/CV-final-project/data/icml_face_data.csv') # CHANGE ME 
    base_filename = "/Users/Natalie/Desktop/cs1430/CV-final-project/data/train/" # CHANGE ME
    for index, row in df.iterrows():
        if (row[' Usage'] == "Training"):
            px = row[' pixels']
            emot = int(row['emotion'])
            emot_loc = emotion_dict[emot]
            filename = base_filename + emot_loc
            img = createPixelArray(px)
            img_arr = augmentIMG(img, task)
            idx = 0
            for i in img_arr:
                num = str(index) + "_" + str(idx)
                idx +=1
                saveIMG(i, num, filename)
def createTest(emotion_dict , task):
    df = pd.read_csv('/Users/Natalie/Desktop/cs1430/CV-final-project/data/icml_face_data.csv') # CHANGE ME
    base_filename = "/Users/Natalie/Desktop/cs1430/CV-final-project/data/test/" # CHANGE ME 
    for index, row in df.iterrows():
        if (row[' Usage'] == "PublicTest"):
            px = row[' pixels']
            emot = int(row['emotion'])
            emot_loc = emotion_dict[emot]
            filename = base_filename + emot_loc
            img = createPixelArray(px)
            img_arr = augmentIMG(img, task)
            idx = 0
            for i in img_arr:
                num = str(index) + "_" + str(idx)
                idx +=1
                saveIMG(i, num, filename)
def createEmotionDict():
    emotionDict = {}
    emotionDict[0]="angry/"
    emotionDict[1]="disgust/"
    emotionDict[2]="fear/"
    emotionDict[3]="happy/"
    emotionDict[4]="sad/"
    emotionDict[5]="surprise/"
    emotionDict[6] = "neutral/"
    return emotionDict
def createSimpleData():
    cleanAll()
    print("Cleaning d")
    emot_dict = createEmotionDict()
    createTrain(emot_dict, 1)
    createTest(emot_dict, 1)
def createComplexData():
    cleanAll()
    emot_dict = createEmotionDict()
    createTrain(emot_dict, 3)
    createTest(emot_dict, 3)
def main():
    emot_dict = createEmotionDict()
    


if __name__ == '__main__':
    main()