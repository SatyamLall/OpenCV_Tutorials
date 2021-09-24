import cv2
import numpy as np

img = cv2.imread("Resources/warpSample.jpeg")

# We can get the pixel points using paint
# Save all points in a float array

# standard dimensions of a playing card
width, height = 250, 350

# pts1 -> actual pixel pts of the part of the image to be extraced
pts1 = np.float32([[325, 444], [559, 347], [462, 813], [712, 712]])
# pts2 -> the mapped pixel pts of the actual extracted image
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

# Creating a matrix using pts1, pts2 and then applying warpPerspective to transform it
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOutput = cv2.warpPerspective(img, matrix, (width, height))

cv2.imshow("Original image", img)
cv2.imshow("Transformed image", imgOutput)
cv2.waitKey(0)
