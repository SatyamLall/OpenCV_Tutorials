import cv2

# Reading image
img = cv2.imread("Resources/sample1.jpeg")

# We convert the image to grayscale because we are basically just looking at brightness (patterns of brightness that define a face)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier("Resources/haarcascades/haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)

# Loop through all the faces
for f in faces:
    x = f[0]
    y = f[1]
    w = f[2]
    h = f[3]

    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 50), 3)

# Displaying image
cv2.imshow("Result", img)
cv2.waitKey(0)