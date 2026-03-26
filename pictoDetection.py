import os
import numpy as np
import cv2
from keras.models import load_model

threshold = 0.50
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
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img = img/255
    return img

while True:
    success, imgOriginal = cap.read()

    if not success:
        continue

    img = np.asarray(imgOriginal)

    cv2.rectangle(img, (100,100), (300,300), (50,50,255), 2)
    crop_img = img[100:300, 100:300]

    img = cv2.resize(crop_img, (64,64))
    img = preprocessing(img)
    img = img.reshape(1, 64, 64, 1)


    cv2.putText(imgOriginal, "Class", (20,35), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(imgOriginal, "Probability", (20,75), font, 0.75, (255,0,255), 2, cv2.LINE_AA)

    prediction = model.predict(img, verbose = 0)

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


