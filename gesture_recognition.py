"""
Gesture recognition built on top of hand_tracking.py.

Detects:
  1. A snap (thumb + middle finger).
  2. Open-closed-open hand as a single action ("from open, make a fist, then
     open it again"). Open is the default reset state, so the gesture only fires
     when the hand deliberately starts open, closes, and reopens.

See GESTURES.md for a full explanation of how every part works.

Run:  python gesture_recognition.py   (press 'q' to quit)
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import math

import system_tools

# ----------------------------------------------------------------------------
# Landmark indices (see GESTURES.md, Section 1)
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# Geometry helpers (see GESTURES.md, Section 2)
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# The detector: holds memory across frames (see GESTURES.md, Sections 4 & 5)
# ----------------------------------------------------------------------------
class GestureDetector:
    # ---- tunable thresholds (see GESTURES.md, Section 7) ----
    # Snap detection is tuned slightly loose: we'd rather catch a real snap than
    # demand a perfect one. A fist (open-closed-open) is still kept distinct via
    # the hold requirement below, so the two never collide.
    PINCH_ON = 0.4          # thumb/middle distance (in palm units) to "arm" a snap
    PINCH_OFF = 0.45        # distance to confirm the fingers separated again
    SNAP_SPEED = 7.0        # middle-fingertip speed (palm-units / second) to fire
    SNAP_ARM_WINDOW = 0.45  # seconds: max gap between pinch and flick
    SNAP_COOLDOWN = 0.6     # seconds: min gap between two snaps
    # A real snap pinches thumb+middle while the hand is open, then the middle
    # finger flicks down -- so the finger count collapses (typically 2 -> 1 or 0)
    # and the tips spring apart. A slowly-closing fist keeps thumb+middle together
    # (it never "releases"), which is what now separates the two.
    SNAP_ARM_FINGERS = 2    # hand must be at least this open to arm a snap
    OPEN_WINDOW = 2.0       # seconds: max gap between each step of open-closed-open
    # Open is the standard resting state and triggers nothing on its own. The
    # open-closed-open action only fires when the hand is deliberately held in a
    # fist before reopening. A snap collapses the finger count for only an
    # instant, so requiring a real hold is what keeps a snap from also firing
    # open-closed-open (which is how they used to be confused).
    OPEN_CLOSED_OPEN_MIN_HOLD = 0.35  # seconds: fist must be held this long

    def __init__(self):
        # open -> closed -> open state machine.
        # phase: "idle" (haven't seen an open hand yet) -> "open" -> "closed" ->
        # back to "open" fires. Starting in "idle" (not "open") means a hand that
        # begins already closed can't fire on its first open: a plain closed->open
        # is ignored, only a real open->closed->open counts.
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
        """Run both detectors on one hand's landmarks for one frame."""
        now = timestamp_ms / 1000.0  # seconds
        scale = palm_size(lm)

        # --- DEBUG: uncomment to watch live numbers while tuning ---
        # pinch_dbg = dist(lm[THUMB_TIP], lm[MIDDLE_TIP]) / scale
        # print(f"state={hand_state(lm):6} pinch={pinch_dbg:.2f}")

        self._detect_open_closed_open(lm, now)
        self._detect_snap(lm, now, scale)

    # ----- Gesture B: open -> closed -> open -----
    def _detect_open_closed_open(self, lm, now):
        state = hand_state(lm)

        # An abandoned fist (held past the window) falls back to "idle", so the
        # next open has to start a fresh open->closed->open from scratch.
        if self.phase == "closed" and now - self.phase_time > self.OPEN_WINDOW:
            self.phase = "idle"
            self.phase_time = now

        if state == "open":
            if self.phase == "closed":
                # open -> closed -> open completed, but only fire if the fist was
                # held long enough to be deliberate. A snap collapses the count
                # for just an instant, so it falls under this hold and is ignored.
                held = now - self.phase_time
                if held >= self.OPEN_CLOSED_OPEN_MIN_HOLD:
                    self.open_closed_open_count += 1
                    audio_state = system_tools.toggle_mute()
                    self._flash(f"OPEN-CLOSED-OPEN! ({audio_state})", now)
            # We're now resting in the open reset state, ready to arm a close.
            self.phase = "open"
            self.phase_time = now
        elif state == "closed":
            # Only advance to "closed" from a genuinely observed "open". From
            # "idle" (hand started closed) we ignore it -- closed->open is not
            # the gesture, only open->closed->open is.
            if self.phase == "open":
                self.phase = "closed"
                self.phase_time = now
        # "other" is a transition pose: hold the current phase.

    # ----- Gesture A: snap -----
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
                # The snap signature: the middle finger flicked down, so the
                # finger count dropped from where we armed. We no longer require a
                # full fist (count <= 1) -- a real snap often keeps index/ring/
                # pinky extended, so demanding a fist was missing most snaps.
                collapsed = count < self.armed_count

                # A slow fist also drops the count, but it never "releases"
                # (thumb+middle stay together) and never reaches snap speed, so
                # the speed + release checks keep snaps and fists distinct.
                if (recently_armed and speed > self.SNAP_SPEED and released
                        and off_cooldown and collapsed):
                    self.snap_count += 1
                    self.last_snap_time = now
                    self.armed = False
                    screen_state = system_tools.toggle_screen()
                    self._flash(f"SNAP! ({screen_state})", now)

        self.prev_mid = mid
        self.prev_time = now

    # ----- overlay text for the UI -----
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


# ----------------------------------------------------------------------------
# MediaPipe plumbing (same as hand_tracking.py, plus the detector hook)
# ----------------------------------------------------------------------------
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
