# Using haarcascades for facial recognition

import cv2

# Reading pre-trained file for facial recognition
faceCascade = cv2.CascadeClassifier("Resources/haarcascades/haarcascade_frontalface_default.xml")

# Using webcame
cap = cv2.VideoCapture(0)
# setting the width -> id no: 3
cap.set(3, 640)
# setting the height -> id no: 4
cap.set(4, 480)
# setting the brightness -> id no: 10
cap.set(10, 100)

while True:
    success, img = cap.read()
    cv2.imshow("Webcam Feed", img)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Getting all the images in frame
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)

    # Iterating through all the faces and getting the co-ordinates for creating a rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 50), 3)

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
