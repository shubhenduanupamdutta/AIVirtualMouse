"""
OpenCV Hand Tracking Min
"""

import cv2 as cv
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode: bool = False, max_hands: int = 2,
                 model_complexity: float = 1,
                 detection_confidence: float = 0.5,
                 track_confidence: float = 0.5):
        self.mode = mode
        self.max = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence
        self.model_complexity = model_complexity
        self.mpHands = mp.solutions.hands  # type: ignore
        self.hands = self.mpHands.Hands(self.mode, self.max,
                                        self.model_complexity,
                                        self.detection_confidence,
                                        self.track_confidence)
        self.mpDraw = mp.solutions.drawing_utils  # type: ignore
        self.results = None
        self.tip_ids = [4, 8, 12, 16, 20]
        self.landmarks = None

    def find_hands(self, img, draw: bool = True):
        drawing_spec = self.mpDraw.DrawingSpec(
            color=(0, 255, 0), thickness=1, circle_radius=1)
        img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(img_RGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        handLandmarks,
                        self.mpHands.HAND_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec
                    )
        return img

    def find_position(self, image, hand_number: int = 0, draw: bool = True):
        landmark_list = []
        if self.results.multi_hand_landmarks:  # type: ignore
            # type: ignore
            my_hand = self.results.multi_hand_landmarks[hand_number]  # type: ignore # noqa E501
            for id, lm in enumerate(my_hand.landmark):
                height, width, channels = image.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                landmark_list.append([id, cx, cy])
                if draw:
                    cv.circle(image, (cx, cy), 7, (255, 0, 0), cv.FILLED)
        self.landmarks = landmark_list
        return landmark_list

    def fingers_up(self):
        # Thumb is ignored
        fingers = []
        tip_ids = self.tip_ids
        for tip_id in tip_ids[1:]:
            if self.landmarks[tip_id][2] < self.landmarks[tip_id - 2][2]:  # type: ignore # noqa E501
                fingers.append(1)  # 1 if open
            else:
                fingers.append(0)

        return fingers


def main():
    previous_time = 0
    current_time = 0
    cap = cv.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        landmarks = detector.find_position(img)
        if len(landmarks) != 0:
            print(landmarks[4])
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv.putText(img, str(int(fps)), (10, 70),
                   cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv.imshow("Image", img)
        key = cv.waitKey(1)
        if key == ord("q"):
            break


if __name__ == '__main__':
    main()
