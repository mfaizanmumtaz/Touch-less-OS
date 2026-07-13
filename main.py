"""Webcam hand-gesture -> app launcher and system control.

Static gestures (single hand, hold steady to trigger once):
  open_palm    -> Chrome
  fist         -> VS Code
  peace        -> File Explorer
  thumbs_up    -> Calculator
  square_frame -> ChatGPT (both hands forming a square/photo-frame shape)

Continuous gestures (pinch thumb+index, other fingers curled):
  Left hand pinch  -> volume (spread apart = louder, together = quieter)
  Right hand pinch  -> brightness (spread apart = brighter, together = dimmer)

Hold a static gesture steady for HOLD_FRAMES consecutive frames to trigger
its action once; the same gesture won't re-trigger until you change gesture.
A brief drop in tracking (up to MISS_TOLERANCE_FRAMES) does not reset the
hold, to absorb normal MediaPipe flicker.
"""

import time

import cv2
import mediapipe as mp

from actions import run_action
from gestures import classify_gesture, is_pinch_pose, is_square_frame, pinch_level
import system_control

HOLD_FRAMES = 15  # ~0.5s at 30fps, avoids accidental triggers on transition
COOLDOWN_SECONDS = 2.0
MISS_TOLERANCE_FRAMES = 3  # short flicker tolerance before a hold resets

# Exponential smoothing factor for pinch level (0..1); lower = smoother/slower.
PINCH_SMOOTHING = 0.3

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def _handle_pinch_control(hand_landmarks_list, handedness_labels, smoothed_levels):
    """Adjusts volume (left hand) / brightness (right hand) from pinch
    distance. Returns a status label for on-screen display, or None."""
    status = None
    for landmarks, handedness_label in zip(hand_landmarks_list, handedness_labels):
        if not is_pinch_pose(landmarks):
            continue
        level = pinch_level(landmarks)
        if level is None:
            continue

        # mediapipe's handedness label is the anatomical hand; the frame is
        # already mirrored (cv2.flip) so this matches what the user sees.
        key = handedness_label
        prev = smoothed_levels.get(key)
        smoothed = level if prev is None else prev + PINCH_SMOOTHING * (level - prev)
        smoothed_levels[key] = smoothed

        if key == "Left":
            system_control.set_volume(smoothed)
            status = f"Volume: {int(smoothed * 100)}%"
        elif key == "Right":
            system_control.set_brightness(smoothed)
            status = f"Brightness: {int(smoothed * 100)}%"
    return status


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    last_gesture = None
    hold_count = 0
    miss_count = 0
    last_trigger_time = 0.0
    last_launched_label = ""
    smoothed_levels = {}

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
            control_status = None
            if results.multi_hand_landmarks and results.multi_handedness:
                hands_landmarks = [h.landmark for h in results.multi_hand_landmarks]
                handedness_labels = [
                    h.classification[0].label for h in results.multi_handedness
                ]

                if len(hands_landmarks) == 2 and is_square_frame(*hands_landmarks):
                    current_gesture = "square_frame"
                else:
                    control_status = _handle_pinch_control(
                        hands_landmarks, handedness_labels, smoothed_levels
                    )
                    if control_status is None:
                        current_gesture = classify_gesture(
                            hands_landmarks[0], handedness_labels[0]
                        )

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            if current_gesture is not None and current_gesture == last_gesture:
                hold_count += 1
                miss_count = 0
                last_gesture = current_gesture
            elif current_gesture is None and last_gesture is not None and miss_count < MISS_TOLERANCE_FRAMES:
                # Brief tracking flicker -- keep the hold alive, don't advance it.
                miss_count += 1
            else:
                hold_count = 0
                miss_count = 0
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

            label = control_status or current_gesture or "no gesture"
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
