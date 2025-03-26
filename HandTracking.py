import cv2
import mediapipe as mp
import time

# Create a VideoCapture object to access the webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    if not success:
        print("Failed to capture image")
        break

    cv2.imshow("Image", img)
    cv2.waitKey(1)

