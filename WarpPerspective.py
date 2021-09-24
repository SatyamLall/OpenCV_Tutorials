import cv2
import numpy as np

img = cv2.imread("Resources/warpSample.jpeg")

# We can get the pixel points using paint
# Save all points in a float array

width, height = 250, 350
pts1 = np.float32([[325, 444], [559, 347], [462, 813], [712, 712]])
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOutput = cv2.warpPerspective(img, matrix, (width, height))

cv2.imshow("Original image", img)
cv2.imshow("Transformed image", imgOutput)
cv2.waitKey(0)
