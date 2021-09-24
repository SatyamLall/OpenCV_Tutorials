import cv2
import numpy as np

# Creating a matrix with 0s -> 0 means black
# Since we havent mentioned the dimension this is a grayscale image
img = np.zeros((512, 512))

# Dimensionality 3 means rgb image -> gives colour functionality
img2 = np.zeros((512,512,3), np.uint8)

# This makes the whole image blue as OpenCV follows BGR convention
img2[:] = 255, 0, 0

# Shape shows the dimensionality of the image -> gives height, width
print(img.shape)
print(img2.shape)

# Creating shapes

# Creating a line -> cv2.line( image_name, starting_pts, end_pts, color, thickness(optional) )
shape = img2.shape #taking height and width
cv2.line(img2, (0,0), (shape[1], shape[0]), (0,255,0), 5)

# Creating a rectangle -> cv2.rectangle( img_name, start_pts, diag_end_pts, color, thickness)
cv2.rectangle(img2, (0, 0), (300, 300), (0, 255, 255), cv2.FILLED)

# Creating a circle -> (img_name, center, radius, color, thick, line)
cv2.circle(img2, (200, 200), 100, (0,244, 123), thickness=5, lineType=None)

# Putting texts
# def putText(img: Any,
#             text: Any,
#             org: Any,
#             fontFace: Any,
#             fontScale: Any,
#             color: Any,
#             thickness: Any = None,
#             lineType: Any = None,
#             bottomLeftOrigin: Any = None) -> None

cv2.putText(img2, "RandomText", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,255), thickness=None)
# cv2.imshow("Image", img)
cv2.imshow("Image2", img2)
cv2.waitKey(0)