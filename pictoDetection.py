import os
import numpy as np
import cv2
from keras.models import load_model

threshold = 0.70
path = 'pictogramsOriginal'
name_classes = sorted(os.listdir(path))

##########
#Initialize camera and model keras
cap = cv2.VideoCapture(2)
font = cv2.FONT_HERSHEY_DUPLEX
model = load_model('PictoTrainingModel.keras')

def get_className(classNo):
    return name_classes[classNo]


def preprocessing(img):
    img = img.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = img / 255.0
    return img

while True:
    success, imgOriginal = cap.read()

    if not success:
        continue

    crop_img = imgOriginal[100:300, 100:300].copy()

    cv2.rectangle(imgOriginal, (100,100), (300,300), (50,50,255), 2)

    img_keras = cv2.resize(crop_img, (64,64))
    img_keras = preprocessing(img_keras)
    img_keras = img_keras.reshape(1, 64, 64, 1)

    cv2.putText(imgOriginal, "Class", (20,35), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(imgOriginal, "Probability", (20,75), font, 0.75, (255,0,255), 2, cv2.LINE_AA)

    prediction = model.predict(img_keras, verbose = 0)

    classIndex = np.argmax(prediction)
    probabilityValue = np.amax(prediction)

    if probabilityValue > threshold:
        namePict = get_className(classIndex)
        text_class = f"{namePict}"

        cv2.putText(imgOriginal, text_class, (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, f"{round(probabilityValue * 100, 2)}%", (180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Result", imgOriginal)
    cv2.imshow("Cropped_Img", crop_img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release ()
cv2.destroyAllWindows()


