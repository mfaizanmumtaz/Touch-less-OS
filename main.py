"""Webcam hand-gesture -> app launcher.

Gestures:
  open_palm    -> Chrome
  fist         -> VS Code
  peace        -> File Explorer
  thumbs_up    -> Calculator
  square_frame -> ChatGPT (both hands forming a square/photo-frame shape)

Hold a gesture steady for HOLD_FRAMES consecutive frames to trigger its
action once; the same gesture won't re-trigger until you change gesture.
"""

import time

import cv2
import mediapipe as mp

from actions import run_action
from gestures import classify_gesture, is_square_frame

HOLD_FRAMES = 15  # ~0.5s at 30fps, avoids accidental triggers on transition
COOLDOWN_SECONDS = 2.0

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    last_gesture = None
    hold_count = 0
    last_trigger_time = 0.0
    last_launched_label = ""

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            current_gesture = None
            if results.multi_hand_landmarks and results.multi_handedness:
                hands_landmarks = [h.landmark for h in results.multi_hand_landmarks]

                if len(hands_landmarks) == 2 and is_square_frame(*hands_landmarks):
                    current_gesture = "square_frame"
                else:
                    landmarks = hands_landmarks[0]
                    handedness_label = results.multi_handedness[0].classification[0].label
                    current_gesture = classify_gesture(landmarks, handedness_label)

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            if current_gesture is not None and current_gesture == last_gesture:
                hold_count += 1
            else:
                hold_count = 0
            last_gesture = current_gesture

            now = time.time()
            if (
                current_gesture is not None
                and hold_count == HOLD_FRAMES
                and (now - last_trigger_time) > COOLDOWN_SECONDS
            ):
                app_name = run_action(current_gesture)
                if app_name:
                    last_launched_label = f"Launched: {app_name}"
                    last_trigger_time = now

            label = current_gesture or "no gesture"
            cv2.putText(
                frame, f"Gesture: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            if last_launched_label:
                cv2.putText(
                    frame, last_launched_label, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
                )

            cv2.imshow("Gesture Control (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
