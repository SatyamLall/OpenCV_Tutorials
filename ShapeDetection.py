import cv2
import numpy as np

# ================================= SHAPE DETECTION ======================================
# 1. Convert to grayscale and blur the image (this simplifies the shape detection process as it reduces the amount of information that has to be processed)
# 2. Use canny edge detector
# 3. Get contours using the openCV findContours function and then use drawContours to display them
# 4. Loop through the contours and find area, perimter, and approximate number of corner pts
# 5. Using number of corner pts, we can detect shapes -> use putText to display the shapes


# Custom function to stack images
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


# Reading the image
imgPath = "Resources/shapeDetectionImg.jpeg"
img = cv2.imread(imgPath)
imgCopy = img.copy()


# Function for detecting contours
def getContours(imgSample):

    # Third and fourth step

    # contours stores all the detected contours
    # cv2.findContours(img_src, retrieval_method, approximation)
    contours, hierarchy = cv2.findContours(imgSample, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Loop through all the contours
    for c in contours:
        area = cv2.contourArea(c)   # find area of each contour
        print(area)
        # Draw the contour on the image copy
        # set ContourIndex = -1 for drawing all contours, select specific index to draw a particular point on the contour

        # we can add an if statement to prevent detecting any noise contours -> this draws contours only for areas > a threshold value
        if area > 300:
            cv2.drawContours(imgCopy, c, -1, (0, 0, 255), 3)

        # Calculate curve length and then approximate corners

        peri = cv2.arcLength(c, True)
        print(peri)

        # find out approximately all corner points
        approxCorners = cv2.approxPolyDP(c, 0.02*peri, True)

        # length of approx gives us number of corners which will help us detect shape
        objCorners = len(approxCorners)
        print(objCorners)

        # find out approx. x and y values and height and width of the shapes
        x, y, w, h = cv2.boundingRect(approxCorners)

        # Fifth step
        if objCorners == 3:
            objType = "Tri"
        elif objCorners == 4:
            aspRatio = w/float(h)
            if aspRatio > 0.95 and aspRatio < 1.05:
                objType = "Square"
            else:
                objType = "Rectangle"
        elif objCorners == 5:
            objType = "Penta"
        elif objCorners == 6:
            objType = "Hexa"
        else:
            objType = "Ellipse"

        # this creates a bounded box around our shapes
        # starting point is x,y and ending point is x+width, y+height
        cv2.rectangle(imgCopy, (x,y), (x+w, y+h), (255, 0, 0), 1)

        # print text of shape
        cv2.putText(imgCopy, objType, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 1)

# First step
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)

# Second step
imgCanny = cv2.Canny(imgGray, 10, 10)
imgCanny2 = cv2.Canny(imgBlur, 10, 10)
# function call to custom function
getContours(imgCanny2)

# Displaying the image
imgBlank = np.zeros_like(img)
imgStack = stackImages(1, ([img, imgGray, imgBlur], [imgCanny, imgCanny2, imgCopy]))
cv2.imshow("ImageStack", imgStack)
cv2.waitKey(0)