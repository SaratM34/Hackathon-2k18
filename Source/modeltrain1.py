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

df = pd.read_csv("C:/Users/user/Desktop/hackathon/src/Book1.csv")

le = preprocessing.LabelEncoder()
le.fit(df["VehicleModel"])
VM_trans = le.transform(df["VehicleModel"])
le.fit(df["VehicleMake"])
VMake_trans = le.transform(df["VehicleMake"])
df.insert(loc=0, column='VM_trans', value=VM_trans)
df.insert(loc=0, column='VMake_trans', value=VMake_trans)


df1 = df[['VehicleYear','VM_trans','VMake_trans']]
y = df['ValuedPrice']


clf = SVC(kernel='rbf')
clf.fit(df1,y)
joblib.dump(clf, 'model.pkl')
test =[[2006,75,13]]
ypred = clf.predict(test)
print(ypred)
