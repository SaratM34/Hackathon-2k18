import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from imutils import paths
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from sklearn.externals import joblib
import argparse
import random
import cv2
import os
import PIL
import matplotlib.pyplot as plt

%matplotlib inline


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
args = "data"


imagePaths = sorted(list(paths.list_images(args)))
random.seed(42)
random.shuffle(imagePaths)
data =[]

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    data.append(image)



ndata = np.asarray(data)
data_flat = ndata.reshape(ndata.shape[0],-1)
df = pd.read_csv("C:/Users/user/Desktop/hackathon/src/Book1.csv")


clf = SVC(kernel="rbf")
clf.fit(data_flat,df["ValuedPrice"])
joblib.dump(clf,'imgmodel.pkl')
img_pred = clf.predict(data_flat)
