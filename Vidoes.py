import cv2

cap = cv2.VideoCapture("Resources/test_video.mp4")

while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    
    # q closes the video if it hasnt finished playing already
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break