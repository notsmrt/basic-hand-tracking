import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import math

import system_tools


WRIST = 0
THUMB_TIP = 4
MIDDLE_MCP = 9
MIDDLE_TIP = 12

# tip vs pip joint for each of the four fingers (thumb handled separately)
FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky finger
    (0, 17),                               # Palm base
]

def dist(a, b):
    """2D distance between two normalized landmarks."""
    return math.hypot(a.x - b.x, a.y - b.y)


def palm_size(lm):
    """Internal 'ruler': wrist to base of middle finger. Used to scale everything."""
    return dist(lm[WRIST], lm[MIDDLE_MCP]) or 1e-6  # avoid divide-by-zero


def fingers_extended(lm):
    """Return (count, dict) of which of the 4 fingers are extended.

    A finger is extended when its tip is farther from the wrist than its PIP joint.
    """
    wrist = lm[WRIST]
    states = {}
    count = 0
    for name in FINGER_TIPS:
        tip = lm[FINGER_TIPS[name]]
        pip = lm[FINGER_PIPS[name]]
        extended = dist(wrist, tip) > dist(wrist, pip)
        states[name] = extended
        count += 1 if extended else 0
    return count, states


def hand_state(lm):
    """Classify the hand pose this frame: 'open', 'closed', or 'other'."""
    count, _ = fingers_extended(lm)
    if count <= 1:
        return "closed"
    if count >= 4:
        return "open"
    return "other"


class GestureDetector:
    PINCH_ON = 0.4          # thumb/middle distance (in palm units) to "arm" a snap
    PINCH_OFF = 0.45        # distance to confirm the fingers separated again
    SNAP_SPEED = 7.0        # middle-fingertip speed (palm-units / second) to fire
    SNAP_ARM_WINDOW = 0.45  # seconds: max gap between pinch and flick
    SNAP_COOLDOWN = 0.6
    SNAP_ARM_FINGERS = 2    # hand must be at least this open to arm a snap
    OPEN_WINDOW = 2.0     
    OPEN_CLOSED_OPEN_MIN_HOLD = 0.35  # seconds: fist must be held this long

    def __init__(self):
        self.phase = "idle"
        self.phase_time = 0.0
        self.open_closed_open_count = 0

        # snap state machine
        self.armed = False
        self.armed_time = 0.0
        self.armed_count = 0        # finger count captured when the snap armed
        self.prev_mid = None        # previous middle-tip (x, y)
        self.prev_time = None
        self.snap_count = 0
        self.last_snap_time = 0.0

        # transient on-screen message
        self.message = ""
        self.message_time = 0.0

    def _flash(self, text, now):
        self.message = text
        self.message_time = now

    def process(self, lm, timestamp_ms):
        now = timestamp_ms / 1000.0  # seconds
        scale = palm_size(lm)

        self._detect_open_closed_open(lm, now)
        self._detect_snap(lm, now, scale)

    def _detect_open_closed_open(self, lm, now):
        state = hand_state(lm)
        if self.phase == "closed" and now - self.phase_time > self.OPEN_WINDOW:
            self.phase = "idle"
            self.phase_time = now

        if state == "open":
            if self.phase == "closed":
                held = now - self.phase_time
                if held >= self.OPEN_CLOSED_OPEN_MIN_HOLD:
                    self.open_closed_open_count += 1
                    audio_state = system_tools.toggle_mute()
                    self._flash(f"OPEN-CLOSED-OPEN! ({audio_state})", now)
            # We're now resting in the open reset state, ready to arm a close.
            self.phase = "open"
            self.phase_time = now
        elif state == "closed":

            if self.phase == "open":
                self.phase = "closed"
                self.phase_time = now

    def _detect_snap(self, lm, now, scale):
        thumb_tip = lm[THUMB_TIP]
        middle_tip = lm[MIDDLE_TIP]
        pinch = dist(thumb_tip, middle_tip) / scale

        count, _ = fingers_extended(lm)

        # Stage 1: arm when the tips pinch while the hand is still open.
        # We remember how open it was so we can confirm the collapse later.
        if pinch < self.PINCH_ON and count >= self.SNAP_ARM_FINGERS:
            self.armed = True
            self.armed_time = now
            self.armed_count = count

        # Stage 2: measure middle-fingertip speed and look for the flick
        mid = (middle_tip.x, middle_tip.y)
        if self.prev_mid is not None and self.prev_time is not None:
            dt = now - self.prev_time
            if dt > 0:
                moved = math.hypot(mid[0] - self.prev_mid[0],
                                   mid[1] - self.prev_mid[1]) / scale
                speed = moved / dt

                recently_armed = self.armed and (now - self.armed_time <= self.SNAP_ARM_WINDOW)
                released = pinch > self.PINCH_OFF
                off_cooldown = now - self.last_snap_time > self.SNAP_COOLDOWN


                collapsed = count < self.armed_count

                if (recently_armed and speed > self.SNAP_SPEED and released
                        and off_cooldown and collapsed):
                    self.snap_count += 1
                    self.last_snap_time = now
                    self.armed = False
                    screen_state = system_tools.toggle_screen()
                    self._flash(f"SNAP! ({screen_state})", now)

        self.prev_mid = mid
        self.prev_time = now

    def overlay_lines(self, lm, now):
        count, _ = fingers_extended(lm)
        lines = [
            f"fingers: {count}   state: {hand_state(lm)}   phase: {self.phase}",
            f"open-closed-open: {self.open_closed_open_count}",
            f"snaps: {self.snap_count}",
        ]
        if self.message and now - self.message_time < 1.0:  # flash for 1 second
            lines.append(self.message)
        return lines

cv2.namedWindow("Gesture Recognition", cv2.WINDOW_NORMAL)

latest_result = None
detector = GestureDetector()
last_timestamp_ms = 0


def on_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result, last_timestamp_ms
    latest_result = result
    last_timestamp_ms = timestamp_ms
    # Run gesture detection on the first detected hand (using MediaPipe's clean timestamp).
    if result and result.hand_landmarks:
        detector.process(result.hand_landmarks[0], timestamp_ms)


def main():
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.4,
        result_callback=on_result,
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame = cv2.flip(frame, 1)  # selfie view
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp)

            annotated = np.copy(frame)
            h, w = annotated.shape[:2]

            if latest_result and latest_result.hand_landmarks:
                for hand_landmarks in latest_result.hand_landmarks:
                    for landmark in hand_landmarks:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)
                    for start_idx, end_idx in HAND_CONNECTIONS:
                        start = (int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h))
                        end = (int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h))
                        cv2.line(annotated, start, end, (255, 0, 0), 2)

                # draw the gesture overlay for the first hand
                now = last_timestamp_ms / 1000.0
                for i, line in enumerate(detector.overlay_lines(latest_result.hand_landmarks[0], now)):
                    color = (0, 255, 255) if line.startswith(("SNAP!", "OPEN-CLOSED-OPEN!")) else (255, 255, 255)
                    cv2.putText(annotated, line, (10, 30 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Gesture Recognition", annotated)

            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
