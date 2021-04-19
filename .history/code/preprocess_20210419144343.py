
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
            createComplexData(data_path)
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

    
    def cleanTestDirs():
        emotions = ['angry', 'happy', 'disgust', 'sad', 'neutral', 'surprise', 'fear']
        for e in emotions:
            pathy = self.data_path+'test/'+e
            pics = 1
            for f in Path(pathy).glob('*.jpg'):
                if (pics <= 100):
                    pics+=1
                else:
                    try:
                    #f.unlink()
                        os.remove(f)
                    except OSError as e:
                        print("Error: %s : %s" % (f, e.strerror))
    def cleanTrainDirs(data_path):
        emotions = ['angry', 'happy', 'disgust', 'sad', 'neutral', 'surprise', 'fear']
        for e in emotions:
            pathy = data_path+'train/'+e
            for f in Path(pathy).glob('*.jpg'):
                try:
                #f.unlink()
                    os.remove(f)
                except OSError as e:
                    print("Error: %s : %s" % (f, e.strerror))
    def cleanAll(data_path):
        cleanTestDirs(data_path)
        cleanTrainDirs(data_path)
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
        img2 = cv2.bilateralFilter(img1, d=9, sigmaColor=75, sigmaSpace=75)
        imgs.append(img2)
        if task == 3:
            kernel = np.array([[-1.0, -1.0, -1.0], 
                    [-1.0, 9, -1.0],
                    [-1.0, -1.0, -1.0]])
            img3 = cv2.filter2D(img2,-1,kernel)
            imgs.append(img3)
            img4 = equalize_hist(img3)
            imgs.append(img4)
            img5 = cv2.bilateralFilter(img4, d=9, sigmaColor=100, sigmaSpace=100)
            imgs.append(img5)
        img6 = cv2.flip(img, 1) # flip horizontally
        imgs.append(img6)
        return imgs
    def saveIMG(arr, num, folderLoc):
        im = Image.fromarray(arr)
        filename = folderLoc + "image_"+ num+".jpg"
        im.save(filename)
    def createTrain(emotion_dict, task, data_path):
        path1 = data_path+"train.csv"
        df = pd.read_csv(path1) # CHANGE ME 
        base_filename = data_path+"train/" # CHANGE ME
        for index, row in df.iterrows():
            px = row['pixels']
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
    def createTest(emotion_dict , task, data_path):
        path1 = data_path +"icml_face_data.csv"
        df = pd.read_csv(path1) # CHANGE ME
        base_filename = data_path + "test/" # CHANGE ME 
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
    def createSimpleData(data_path):
        cleanAll(data_path)
        print("Cleaning done")
        emot_dict = createEmotionDict()
        createTrain(emot_dict, 1, data_path)
        print("Training done")
        createTest(emot_dict, 1, data_path)
        print("Testing done")
    def createComplexData(data_path):
        cleanAll(data_path)
        emot_dict = createEmotionDict()
        createTrain(emot_dict, 3, data_path)
        createTest(emot_dict, 3, data_path)
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