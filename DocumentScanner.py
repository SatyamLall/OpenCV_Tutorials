import cv2
import numpy as np

# Setting dimensions
imgWidth = 480
imgHeight = 640


# Capturing webcam feed
cap = cv2.VideoCapture(0)
cap.set(3, imgWidth)
cap.set(4, imgHeight)
cap.set(10, 150)


# ======================================  Steps  ==========================================
# 1. Preprocessing the image: Convert the image to grayscale, blur it and detect edges
# 2. Detect contours and find the largest contour
# 3. Get the corner points of the largest contour and then use warp perspective to display it

# Custom function to stack images
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor

    return ver


# Function to preprocess the image
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, 100, 100)
    # Dilation is used to make the edges thicker and erode to make it thinner -> helps in better edge detection
    kernel = np.ones((5,5))  # 5 by 5 matrix of ones
    imgDilate = cv2.dilate(imgCanny, kernel, iterations=2)
    imgErode = cv2.erode(imgDilate, kernel, iterations=1)

    return imgErode


# Function to detect contours
def getContours(imgSample):
    contours, hierarchy = cv2.findContours(imgSample, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    maxArea = 0
    maxContour = np.array([])
    for c in contours:
        area = cv2.contourArea(c)
        if area > 3000:
            peri = cv2.arcLength(c, True)
            approxCorners = cv2.approxPolyDP(c, 0.02*peri, True)
            if area > maxArea and len(approxCorners) == 4:
                maxArea = area
                maxContour = approxCorners
    # Draw the biggest contour
    cv2.drawContours(imgContour, maxContour, -1, (255, 255, 0), 20)

    return maxContour


# Function to re-order the corner points of the biggest contour so as to display the image correctly
def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis = 1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


# Function to warp the image
def warp(img, maxContour):
    maxContour = reorder(maxContour)
    pts1 = np.float32(maxContour)
    pts2 = np.float32([[0,0], [imgWidth, 0], [0, imgHeight], [imgWidth, imgHeight]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (imgWidth, imgHeight))
    # Crop a little of the edges
    imgCropped = imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]
    imgCropped = cv2.resize(imgCropped, (imgWidth, imgHeight))

    return imgCropped

# Main code to read the webcam input and call all functions
while True:
    success, img = cap.read()
    img = cv2.resize(img, (imgWidth, imgHeight))
    imgContour = img.copy()

    imgThreshold = preProcess(img)
    maxContour = getContours(imgThreshold)

    if maxContour.size != 0:
        imgWarped = warp(img, maxContour)
        imgArr = ([imgContour, imgWarped])
        cv2.imshow("Result", imgWarped)
    else:
        imgArr = ([imgContour, img])

    imgStack = stackImages(0.6, imgArr)
    cv2.imshow("Workflow", imgStack)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break