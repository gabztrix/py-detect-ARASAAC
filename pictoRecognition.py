import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Activation, Dropout, Flatten
from keras.optimizers import Adam

##########
#Variables
path='pictogramsOriginal'
images=[]
classNo=[]
testRatio=0.2
valRAtio=0.2
imgDimension=(32,32,3)
##########

myList = os.listdir(path)
numClasses=len(myList)
print(f"Number of classes: {numClasses}")

