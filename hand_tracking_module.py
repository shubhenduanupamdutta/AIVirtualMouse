"""
OpenCV Hand Tracking Min
"""
import math

import cv2 as cv
import mediapipe as mp
import time


class HandDetector:
    def __init__(
        self,
        mode: bool = False,
        max_hands: int = 2,
        model_complexity: float = 1,
        detection_confidence: float = 0.5,
        track_confidence: float = 0.5,
    ):
        self.mode = mode
        self.max = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence
        self.model_complexity = model_complexity
        self.mpHands = mp.solutions.hands  # type: ignore
        self.hands = self.mpHands.Hands(
            self.mode,
            self.max,
            self.model_complexity,
            self.detection_confidence,
            self.track_confidence,
        )
        self.mpDraw = mp.solutions.drawing_utils  # type: ignore
        self.results = None
        self.tip_ids = [4, 8, 12, 16, 20]
        self.landmarks = None

    def find_hands(self, img, draw: bool = True):
        drawing_spec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
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
                        connection_drawing_spec=drawing_spec,
                    )
        return img

    def find_position(self, image, hand_number: int = 0, draw: bool = True):
        x_list = []
        y_list = []
        bounding_box = []
        self.landmarks = []

        if self.results.multi_hand_landmarks:  # type: ignore
            # type: ignore
            my_hand = self.results.multi_hand_landmarks[hand_number]  # type: ignore # noqa E501
            for id, lm in enumerate(my_hand.landmark):
                height, width, channels = image.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                x_list.append(cx)
                y_list.append(cy)
                self.landmarks.append([id, cx, cy])
                if draw:
                    cv.circle(image, (cx, cy), 7, (255, 0, 0), cv.FILLED)

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            bounding_box = x_min, y_min, x_max, y_max

            if draw:
                cv.rectangle(
                    image, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2
                )
        return self.landmarks, bounding_box

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

    def find_distance(self, p1: int, p2: int, image, draw: bool = True, r: int = 15, t: int = 3):
        x1, y1 = self.landmarks[p1][1:]
        x2, y2 = self.landmarks[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv.line(image, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(image, (x1, y1), r, (255, 0, 255), cv.FILLED)
            cv.circle(image, (x2, y2), r, (255, 0, 255), cv.FILLED)
            cv.circle(image, (cx, cy), r, (0, 0, 255), cv.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, image, [x1, y1, x2, y2, cx, cy]


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

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv.imshow("Image", img)
        key = cv.waitKey(1)
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
