from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import cv2
from imutils import face_utils
import argparse
import imutils
import dlib
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
def augmentIMG(img):
    imgs = [img]
    img1 = equalize_hist(img)
    imgs.append(img1)
    img2 = cv2.bilateralFilter(img1, d=9, sigmaColor=75, sigmaSpace=75)
    imgs.append(img2)
    img6 = cv2.flip(img, 1) # flip horizontally
    imgs.append(img6)
def saveIMG(arr, num, folderLoc):
    im = Image.fromarray(arr)
    filename = folderLoc + "image_"+ str(num)+".jpg"
    im.save(filename)
def createTrain():
    df = pd.read_csv('/Users/Natalie/Desktop/cs1430/CV-final-project/data/train.csv') 
    for index, row in df.iterrows():
        px = row['pixels']
        emot = row['emotion']
        img = createPixelArray(df.pixels[5])
        
def createEmotionDict():
    emotionDict = {}
    emotionDict[0]="angry/"
    emotionDict[1]="disgust/"
    emotionDict[2]="fear/"
    emotionDict[3]="happy/"
    emotionDict[4]="sad/"
    emotionDict[5]="surprise/"
    emotionDict[6] = "neutral/"
def main():
    df = pd.read_csv('/content/train.csv') 
    img = createPixelArray(df.pixels[5])
    showImages(imgs)


if __name__ == '__main__':
    main()