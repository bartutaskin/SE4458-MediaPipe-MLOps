import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, 
                 mode=False, 
                 maxHands = 2, 
                 detectionConf = 0.5, 
                 trackingConf = 0.5):
        """
        Initializes the hand detector with specified parameters.

        Args:
            mode (bool, optional): If true, processes each frame as a separate image (for static images). Defaults to False.
            maxHands (int, optional): Maximum number of hands to detect. Defaults to 2.
            detectionConf (float, optional): Minimum confidence value for detecting hands. Defaults to 0.5.
            trackingConf (float, optional): Minimum confidence value for tracking hands. Defaults to 0.5.
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, 
                                        max_num_hands=self.maxHands, 
                                        min_detection_confidence=self.detectionConf, 
                                        min_tracking_confidence=self.trackingConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        """
        Finds hands in the image and draws landmarks if specified.

        Args:
            img (numpy.ndarray): Input image.
            draw (bool, optional): If true, draws landmarks on the image. Defaults to True.

        Returns:
            numpy.ndarray: The image with / without landmarks drwan.
        
        Requires:
            img (numpy.ndarray): The input image in BGR format. Should not be none.
        
        Effects:
            Converts the image to RGB format.
            Processes the image to find hands.
            Draws landmarks on the image if draw is true.
            Returns the image with landmarks drawn if draw is true, otherwise returns the original image.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        """
        Finds the position of hand landmarks in the image.

        Args:
            img (numpy.ndarray): The input image to find positions in.
            handNo (int, optional): Specifies which hand to track (0 = first detected hand). Defaults to 0.
            draw (bool, optional): If true, draws circles to landmark points. Defaults to True.

        Returns:
            list: Returns a list of landmark positions in the format [id, x, y].
        
        Requires:
            img (numpy.ndarray): The input image in BGR format. Should not be none.
            findHands: The findHands method should be called before this method to ensure hands are detected.
        
        Effects:
            Finds the positions of hand landmarks in the image.
            Draws circles on the landmarks if draw is true.
            Returns a list of landmark positions in the format [id, x, y].
        """
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
        return lmList

def calculateFPS(pTime, cTime):
    """
    Calculates frames per second (FPS).

    Args:
        pTime (float): The previous time (last frame time).
        cTime (float): The current time (current frame time).

    Returns:
        float: The FPS value.
    """
    return 1 / (cTime - pTime)

def main():    
    """
    Main function to capture video, detect hands, and display FPS.

    Effects:
        Captures vide from the webcam.
        Uses handDetector to detect hands and track landmarks.
        Displays the video with landmarks and FPS on the screen.
        Exits the loop when 'q' is pressed.
    """
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    try:
        while True:
            success, img = cap.read()

            if not success:
                print("Failed to capture image")
                break

            img = detector.findHands(img)
            lmList = detector.findPosition(img)

            # Uncomment the following line to print the landmark positions
            # if lmList:
            #     print(lmList[4]) # Prints the position of thumb (4th landmark)
            
            cTime = time.time()
            fps= calculateFPS(pTime, cTime)
            pTime = cTime

            cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()