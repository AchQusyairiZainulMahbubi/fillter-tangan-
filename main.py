import cv2
import mediapipe as mp
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton
from hand_tracking import (
    THUMB_TIP,
    INDEX_TIP,
    MIDDLE_TIP,
    RING_TIP,
    PINKY_TIP,
    get_extended_fingers,
)
from geometry import render_portal
from filters import FILTER_SETS

def is_fist(fingers):
    return not any(fingers.values())

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Portal", cv2.WINDOW_NORMAL)

     
    if not cap.isOpened():
        raise RuntimeError(
            "No se pudo abrir la camara. Revisa el indice de camara o los permisos."
        )
    filter_index = 0
    gesture_locked = False
     
    

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        left_hand = None
        right_hand = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                raw_label = handedness.classification[0].label
                label = "Right" if raw_label == "Left" else "Left"

                if label == "Left":
                    left_hand = hand_landmarks
                else:
                    right_hand = hand_landmarks

        if left_hand is not None and right_hand is not None:
            lm_left = left_hand.landmark
            lm_right = right_hand.landmark

            left_fingers = get_extended_fingers(left_hand, w, h)
            right_fingers = get_extended_fingers(right_hand, w, h)

            both_fist = is_fist(left_fingers) and is_fist(right_fingers)

            if both_fist and not gesture_locked:
                gesture_locked = True
                filter_index = (filter_index + 1) % len(FILTER_SETS)

            elif not both_fist:
                gesture_locked = False

            thumb_left = (lm_left[THUMB_TIP].x * w, lm_left[THUMB_TIP].y * h)
            index_left = (lm_left[INDEX_TIP].x * w, lm_left[INDEX_TIP].y * h)
            middle_left = (lm_left[MIDDLE_TIP].x * w, lm_left[MIDDLE_TIP].y * h)
            ring_left = (lm_left[RING_TIP].x * w, lm_left[RING_TIP].y * h)
            pinky_left = (
                lm_left[PINKY_TIP].x * w,
                lm_left[PINKY_TIP].y * h,
            )

            thumb_right = (lm_right[THUMB_TIP].x * w, lm_right[THUMB_TIP].y * h)
            index_right = (lm_right[INDEX_TIP].x * w, lm_right[INDEX_TIP].y * h)
            middle_right = (lm_right[MIDDLE_TIP].x * w, lm_right[MIDDLE_TIP].y * h)
            ring_right = (lm_right[RING_TIP].x * w, lm_right[RING_TIP].y * h)
            pinky_right = (
                lm_right[PINKY_TIP].x * w,
                lm_right[PINKY_TIP].y * h,
            )

            filter1, filter2, filter3 = FILTER_SETS[filter_index]
            # Portal 1 (Grid)
            frame = render_portal(
                frame,
                thumb_left,
                index_left,
                thumb_right,
                index_right,
                filter1,
            )

            # Portal 2 (RGB Glitch)
            frame = render_portal(
                frame,
                index_left,
                middle_left,
                index_right,
                middle_right,
                filter2,
            )

            # Portal 3 (Halftone)
            frame = render_portal(
                frame,
                middle_left,
                pinky_left,
                middle_right,
                pinky_right,
                filter3,
            )

        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



    