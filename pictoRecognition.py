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
valRatio=0.2
imgDimension=(32,32,3)
##########

myList = os.listdir(path)
numClasses=len(myList)
print(f"Number of classes: {numClasses}")

##########
# Import, Resize and Augment images

dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10,
    brightness_range=[0.5, 1.5],
    fill_mode='nearest'
)

for class_id, folder_name in enumerate(myList):
    folder_path = os.path.join(path, folder_name)
    myPicList = os.listdir(folder_path)

    for y in myPicList:
        img_path = os.path.join(folder_path, y)
        curImg = cv2.imread(img_path)

        if curImg is not None:
            curImg = cv2.resize(curImg, (imgDimension[0], imgDimension[1]))
            images.append(curImg)
            classNo.append(class_id)

            imgExpanded = np.expand_dims(curImg, axis=0)
            imgVariations = dataGen.flow(imgExpanded, batch_size=1)

            for i in range(50):
                nova_img = next(imgVariations)[0].astype('uint8')
                images.append(nova_img)
                classNo.append(class_id)

    print(f"Classe {class_id} ({folder_name}) carregada.")

images = np.array(images)
classNo = np.array(classNo)

##########
#Spliting the Data

x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size = testRatio)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = valRatio)

numSample = []

for x in range(0, numClasses):
    numSample.append(len(np.where(y_train == x)[0]))

def preprocessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

x_train = np.array(list(map(preprocessing, x_train)))
x_test = np.array(list(map(preprocessing, x_test)))
x_validation = np.array(list(map(preprocessing, x_validation)))

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2],1)

##########
#Image Generation

dataGen.fit(x_train)

y_train = to_categorical(y_train, numClasses)
y_test = to_categorical(y_test, numClasses)
y_validation = to_categorical(y_validation, numClasses)


def myModel():
    noFilters = 60
    sizeFilter1 = (5, 5)
    sizeFilter2 = (3, 3)
    sizePool = (2, 2)
    noNode = 50

    model = Sequential()
    model.add((Conv2D(noFilters, sizeFilter1, input_shape=(imgDimension[0], imgDimension[1], 1), activation='relu')))
    model.add((Conv2D(noFilters, sizeFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizePool))

    model.add(Conv2D(noFilters // 2, sizeFilter2, activation='relu'))
    model.add(Conv2D(noFilters // 2, sizeFilter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizePool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noNode, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numClasses, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())

history = model.fit(
    x_train,
    y_train,
    batch_size=50,
    epochs=50,
    validation_data=(x_validation, y_validation),
    shuffle=True
)

model.save("PictoTrainingModel.keras")
