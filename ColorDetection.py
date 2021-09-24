import cv2
import numpy as np

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


# Function for trackbar
def empty(a):
    pass

imgPath = "Resources/colorDetectionImg.jpeg"

# Colour Detection
#
# Steps:
#
#     1. Convert image to HSV
#     2. Define hue, saturation and value limits -> for this we need to find out HSV values for that particular color, so we use trackbars to do that
#     3. Create trackbars for hsv min, max values -> createTrackbar method
#     4. Read trackbar values and apply to the image -> getTrackbar method
#     5. Create a mask for the image in the range of the trackbar values
#     6. Once we get satisfactory hsv values for the mask, we keep them as the default values
#     7. Create the result by performing the bitwise and operation b/w the mask and the original image

## Step 1: Convert image to HSV
# imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

## Step 2 & 3: Define hue, saturation, value limits and create trackbars

# Create a new window for trackbars and resize it
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 660, 260)

# cv2.createTrackbar(name, windowName, defaultValue, maxVal, func)

cv2.createTrackbar("Hue Min", "Trackbars", 90, 179, empty)
cv2.createTrackbar("Hue Max", "Trackbars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "Trackbars", 95, 255, empty)
cv2.createTrackbar("Sat Max", "Trackbars", 255, 255, empty)
cv2.createTrackbar("Val Min", "Trackbars", 95, 255, empty)
cv2.createTrackbar("Val Max", "Trackbars", 255, 255, empty)

## Step 4: Get trackbar values

while True:

    img = cv2.imread(imgPath)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue Min", "Trackbars")
    h_max = cv2.getTrackbarPos("Hue Max", "Trackbars")
    s_min = cv2.getTrackbarPos("Sat Min", "Trackbars")
    s_max = cv2.getTrackbarPos("Sat Max", "Trackbars")
    v_min = cv2.getTrackbarPos("Val Min", "Trackbars")
    v_max = cv2.getTrackbarPos("Val Max", "Trackbars")

    # print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    ## Step 5: Creating a mask

    mask = cv2.inRange(imgHSV, lower, upper)

    # we get 90, 179, 95, 255, 179, 255 as best values

    ## Step 6 & 7: And mask and hsv image

    imgRes = cv2.bitwise_and(img, img, dst=None, mask=mask)

    # Displaing images

    imgStack = stackImages(0.3, [[img, imgHSV], [mask, imgRes]])
    cv2.imshow("Images", imgStack)
    
    cv2.waitKey(1)
