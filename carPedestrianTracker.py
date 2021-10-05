import cv2

# Initialisation
imgWidth = 640
imgHeight = 480
carCascade = cv2.CascadeClassifier("Resources/cars.xml")
bodyCascade = cv2.CascadeClassifier("Resources/haarcascades/haarcascade_fullbody.xml")
video = cv2.VideoCapture("Resources/dashcamFootage.mp4")

while True:
    success, img = video.read()
    if success:
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        break

    cars = carCascade.detectMultiScale(imgGray)
    bodies = bodyCascade.detectMultiScale(imgGray)

    for (x,y,w,h) in cars:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 10)

    for (x,y,w,h) in bodies:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 5)

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break