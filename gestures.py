"""Static hand-gesture classification from MediaPipe hand landmarks."""

# Landmark indices (MediaPipe Hands)
THUMB_TIP, THUMB_IP = 4, 3
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18
WRIST = 0


def _finger_up(landmarks, tip_idx, pip_idx):
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def _thumb_up(landmarks, handedness_label):
    # Thumb moves sideways rather than up/down; compare x based on hand side.
    if handedness_label == "Right":
        sideways_extended = landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x
    else:
        sideways_extended = landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x

    # A merely-resting thumb can still pass the sideways check above, so also
    # require the tip to be clearly farther from the wrist than the IP joint
    # is -- a resting thumb stays folded in close to the palm.
    wrist = landmarks[WRIST]
    tip_dist = ((landmarks[THUMB_TIP].x - wrist.x) ** 2 + (landmarks[THUMB_TIP].y - wrist.y) ** 2) ** 0.5
    ip_dist = ((landmarks[THUMB_IP].x - wrist.x) ** 2 + (landmarks[THUMB_IP].y - wrist.y) ** 2) ** 0.5
    extended_from_palm = tip_dist > ip_dist * 1.2

    return sideways_extended and extended_from_palm


def _dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def is_square_frame(landmarks_a, landmarks_b):
    """Two-hand 'photo frame' gesture: both thumbs extended and touching tip
    to tip, both index fingers extended and touching tip to tip, other
    fingers curled -- outlines a square/rectangle."""
    for landmarks in (landmarks_a, landmarks_b):
        index_up = _finger_up(landmarks, INDEX_TIP, INDEX_PIP)
        middle_up = _finger_up(landmarks, MIDDLE_TIP, MIDDLE_PIP)
        ring_up = _finger_up(landmarks, RING_TIP, RING_PIP)
        pinky_up = _finger_up(landmarks, PINKY_TIP, PINKY_PIP)
        if not index_up or middle_up or ring_up or pinky_up:
            return False

    # Frame size (distance between the two wrists) sets the scale so the
    # "tips touching" threshold below works at any distance from the camera.
    frame_scale = _dist(landmarks_a[WRIST], landmarks_b[WRIST])
    if frame_scale < 1e-6:
        return False

    thumbs_touch = _dist(landmarks_a[THUMB_TIP], landmarks_b[THUMB_TIP]) < frame_scale * 0.25
    index_touch = _dist(landmarks_a[INDEX_TIP], landmarks_b[INDEX_TIP]) < frame_scale * 0.25

    return thumbs_touch and index_touch


def classify_gesture(landmarks, handedness_label):
    """Returns one of: 'fist', 'open_palm', 'peace', 'thumbs_up', or None."""
    index_up = _finger_up(landmarks, INDEX_TIP, INDEX_PIP)
    middle_up = _finger_up(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    ring_up = _finger_up(landmarks, RING_TIP, RING_PIP)
    pinky_up = _finger_up(landmarks, PINKY_TIP, PINKY_PIP)
    thumb_up = _thumb_up(landmarks, handedness_label)

    fingers_up = [thumb_up, index_up, middle_up, ring_up, pinky_up]

    if all(fingers_up):
        return "open_palm"
    if not any(fingers_up):
        return "fist"
    if index_up and middle_up and not ring_up and not pinky_up:
        return "peace"
    if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "thumbs_up"
    return None
