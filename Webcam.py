import cv2

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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break