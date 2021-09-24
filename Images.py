import cv2

# Reading the image
img = cv2.imread("Resources/sample1.jpeg")

# Resizing
img2 = cv2.resize(img, (430, 500))

# Converting to grayScale
imgGray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Blurring the image -> higher the value more the blurring
imgBlur = cv2.GaussianBlur(img2, (7,7), 0)
imgBlur2 = cv2.GaussianBlur(imgGray, (5, 5), 0)

# Edge detection -> reducing these values detects more edges
imgCanny = cv2.Canny(img2, 100, 100)
imgCanny2 = cv2.Canny(imgBlur, 130, 130)
imgCanny3 = cv2.Canny(imgBlur2, 130, 130)

# Displaying the image
cv2.imshow("Regular Image", img)
cv2.imshow("Sized Image", img2)
cv2.imshow("Gray Image", imgGray)
cv2.imshow("Blurred Image", imgBlur)
cv2.imshow("Canny Image", imgCanny)
cv2.imshow("Canny Image2", imgCanny2)
cv2.imshow("Canny Image3", imgCanny3)

cv2.waitKey(0);

