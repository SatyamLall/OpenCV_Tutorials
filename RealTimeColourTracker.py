import cv2
import numpy as np

frameHeight = 300
frameWidth = 300
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

myColors = [[147, 55, 115, 173, 195, 255],  # pink
            [18, 45, 100, 32, 200, 255]]    # yellow

colorName = [[255,192,203], # pink
             [0,255,255]]   # yellow

# This list is appended with all the points that need to be traced -> [x, y, clrIndex]
myPts = []


def findColor(img, myColors, colorName):

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cnt = 0
    # temp list to append to myPts list
    newPoints = []
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        x, y = getContours(mask)
        cv2.circle(imgResult,(int(x),int(y)), 10, colorName[cnt], cv2.FILLED)
        if x != 0 and y != 0:
            newPoints.append([x,y,cnt])
        cnt += 1

    return newPoints


def getContours(imgSample):

    contours, hierarchy = cv2.findContours(imgSample, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 100:
            cv2.drawContours(imgResult, c, -1, (0, 0, 255), 3)
            peri = cv2.arcLength(c, True)
            approxCorners = cv2.approxPolyDP(c, 0.02*peri, True)
            x, y, w, h = cv2.boundingRect(approxCorners)

    # return center of the contour
    return x+w/2, y


def drawPts(myPts, colorName):
    for point in myPts:
        cv2.circle(imgResult, (int(point[0]), int(point[1])), 10, colorName[point[2]], cv2.FILLED)

while True:
    success, img = cap.read()
    if img is None:
        break
    imgResult = img.copy()
    newPoints = findColor(img, myColors, colorName)
    if len(newPoints) != 0:
        for nPt in newPoints:
            myPts.append(nPt)
    if len(myPts) != 0:
        drawPts(myPts, colorName)

    cv2.imshow("Webcam Feed", imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break