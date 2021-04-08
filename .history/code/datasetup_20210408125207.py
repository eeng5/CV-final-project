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






def main():
    


if __name__ == '__main__':
    main()