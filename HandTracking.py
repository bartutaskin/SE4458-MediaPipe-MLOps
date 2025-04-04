import cv2
import mediapipe as mp
import time
import mlflow
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Hand Tracking with MLflow")
    parser.add_argument('--static_image_mode', type=bool, default=False, help='Whether to treat input images as static.')
    parser.add_argument('--max_num_hands', type=int, default=2, help='Maximum number of hands to detect.')
    parser.add_argument('--min_detection_confidence', type=float, default=0.5, help='Minimum detection confidence.')
    parser.add_argument('--min_tracking_confidence', type=float, default=0.5, help='Minimum tracking confidence.')
    parser.add_argument('--webcam_index', type=int, default=0, help='Webcam index (default is 0).')
    return parser.parse_args()

def main():
    args = parse_args()
    # Set the MLflow experiment name
    mlflow.set_experiment("HandTracking-FPS")

    # Create a VideoCapture object to access the webcam
    cap = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=args.static_image_mode,
        max_num_hands=args.max_num_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )
    mpDraw = mp.solutions.drawing_utils

    pTime = 0
    cTime = 0

    with mlflow.start_run():
        # Log model parameters once at the beginning
        mlflow.log_param("static_image_mode", args.static_image_mode)
        mlflow.log_param("max_num_hands", args.max_num_hands)
        mlflow.log_param("min_detection_confidence", args.min_detection_confidence)
        mlflow.log_param("min_tracking_confidence", args.min_tracking_confidence)
        mlflow.log_param("webcam_index", args.webcam_index)
        
        while True:
            success, img = cap.read()

            if not success:
                print("Failed to capture image")
                break

            # Convert the image to RGB format for MediaPipe
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            #print(results.multi_hand_landmarks)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        #print(id,lm)
                        h, w, c=img.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        #print(id, cx,cy)
                        if id==4:
                            cv2.circle(img, (cx,cy),15,(255,0,255), cv2.FILLED)
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
            cTime = time.time()
            fps= 1/(cTime - pTime)
            mlflow.log_metric("FPS", fps)
            pTime = cTime

            cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

