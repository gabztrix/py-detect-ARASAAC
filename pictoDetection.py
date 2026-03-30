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

#Preprocess image using Otsu Threshold
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

    #Prepare image to find contour
    imgGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)

    #Detect image edges (strong lines)
    imgCanny = cv2.Canny(imgBlur, 50, 150)

    #Find closed contours
    contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    #Analyze the contours found
    for cnt in contours:
        area = cv2.contourArea(cnt)

        #Ignore tiny areas
        if area > 5000:
            #Dynamic crop based on position found
            x, y, w, h = cv2.boundingRect(cnt)
            crop_img = imgOriginal[y:y + h, x:x + w].copy()

            #Prepare cropped image for Keras model
            img_keras = cv2.resize(crop_img, (64, 64))
            img_keras = preprocessing(img_keras)
            img_keras = img_keras.reshape(1, 64, 64, 1)

            #Prediction
            prediction = model.predict(img_keras, verbose=0)
            classIndex = np.argmax(prediction)
            probabilityValue = np.amax(prediction)

            if probabilityValue > threshold:
                namePict = get_className(classIndex)

                cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 3)
                texto_classe = f"{namePict} {round(probabilityValue * 100, 1)}%"
                cv2.putText(imgOriginal, texto_classe, (x, y - 10), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow("Cropped_Img", crop_img)
                break

    cv2.imshow("Result", imgOriginal)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


