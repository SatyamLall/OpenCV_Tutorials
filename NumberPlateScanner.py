import cv2


# Defining parameters
imgWidth = 640
imgHeight = 480
fontColor = (255,155,255)
rectColor = (255,255,0)
minArea = 200
numPlateCascade = cv2.CascadeClassifier("Resources/haarcascades/haarcascade_russian_plate_number.xml")


# Taking webcam input
cap = cv2.VideoCapture(0)
cap.set(3, imgWidth)
cap.set(4, imgHeight)


count = 0   # Counter for storing images

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numPlates = numPlateCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numPlates:
        area = w*h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 50), 3)
            cv2.putText(img, "Number plate detected", (x, y-5), cv2.FONT_ITALIC, 1, fontColor, 4)
            imgCropped = img[y:y+h, x:x+w]
            cv2.imshow("DetectedNumPlate", imgCropped)
    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        count += 1
        cv2.imwrite("Resources/ScannedNumberPlates/NoPlate_"+str(count)+".jpg", imgCropped)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Scan Saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX,
                    2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
