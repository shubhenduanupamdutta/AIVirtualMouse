import cv2 as cv
import numpy as np
import time
import hand_tracking_module as htm
import pyautogui as pag

pag.FAILSAFE = False
# ########## Variables ##########
width_cam, height_cam = 640, 480
# reductions
reduction_x_left = 100
reduction_x_right = 100
reduction_y_top = 20
reduction_y_bottom = 180

SMOOTHENING_VALUE = 5
# ########## Main ##########


cap = cv.VideoCapture(0)
cap.set(3, width_cam)
cap.set(4, height_cam)

previous_time = 0

previous_location_x, previous_location_y = 0, 0
current_location_x, current_location_y = 0, 0

detector = htm.HandDetector(max_hands=1)
screen_width, screen_height = pag.size()
print(screen_width, screen_height)

while True:
    success, image = cap.read()
    image = cv.flip(image, 1)

    # done 1: Find hand landmarks
    image = detector.find_hands(image)
    landmarks, bbox = detector.find_position(image)
    # done 2: Get the tip of the index and middle fingers
    if landmarks:
        x1, y1 = landmarks[8][1:]
        x2, y2 = landmarks[12][1:]
        cv.rectangle(
            image,
            (reduction_x_left, reduction_y_top),
            (width_cam - reduction_x_right, height_cam - reduction_y_bottom),
            (255, 0, 255),
            2,
        )
        # done 3: Check which fingers are up
        fingers = detector.fingers_up()
        # print(fingers)
        # done 4: Only index finger: Moving mode
        if fingers[0] == 1 and fingers[1] == 0:
            # done 5: Convert coordinates
            screen_x = np.interp(
                x1, (reduction_x_left, width_cam - reduction_x_right), (0, screen_width)
            )
            screen_y = np.interp(
                y1, (reduction_y_top, height_cam - reduction_y_bottom), (0, screen_height)
            )

            # done 6: Smoothen values
            current_location_x = (
                previous_location_x + (screen_x - previous_location_x) // SMOOTHENING_VALUE
            )
            current_location_y = (
                previous_location_y + (screen_y - previous_location_y) // SMOOTHENING_VALUE
            )
            # done 7: Move mouse
            pag.moveTo(current_location_x, current_location_y)
            previous_location_x, previous_location_y = current_location_x, current_location_y

            cv.circle(image, (x1, y1), 15, (255, 0, 255), cv.FILLED)

        # done 8: Both index and middle fingers are up: Clicking mode
        if fingers[0] == 1 and fingers[1] == 1:
            # done 9: Find distance between fingers
            length, image, line_info = detector.find_distance(8, 12, image)
            print(length)
            # done 10: Click mouse if distance short
            if length < 40:
                cv.circle(image, (line_info[4], line_info[5]), 15, (0, 255, 0), cv.FILLED)
                pag.click()

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv.putText(image, f"FPS: {int(fps)}", (20, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv.imshow("Image", image)
    key = cv.waitKey(1)
    if key == ord("q"):
        break
