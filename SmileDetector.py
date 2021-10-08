import cv2
import numpy as np

# Initializing parameters
imgWidth = 640
imgHeight = 480
faceFrameColor = (0,255,0)
fontColor = (0,255,129)
faceCascade = cv2.CascadeClassifier("Resources/haarcascades/haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier("Resources/haarcascades/haarcascade_smile.xml")
cap = cv2.VideoCapture(0)

while True:

    success, img = cap.read()

    if success:
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        break

    faces = faceCascade.detectMultiScale(imgGray, 1.2, 3)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)

        # Extract the face from the img and then detect smile on that face
        imgCropped = img[y:y+h, x:x+w]
        imgCroppedGray = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2GRAY)

        # increasing scale factor and min-neighbours helps in more accurate detection
        smiles = smileCascade.detectMultiScale(imgCroppedGray, 1.6, 20)

        if len(smiles) > 0:
            cv2.putText(img, "Smiling", (x,y-10), cv2.FONT_ITALIC, 1, fontColor, 2)

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break