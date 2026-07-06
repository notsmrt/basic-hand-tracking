import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

# Create a display window
cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)

# Global variable to store the latest results
latest_result = None

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky finger
    (0, 17) # Palm base
]

# Callback function to process detection results
def on_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

def main():
    # Initialize MediaPipe Hand Landmarker
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.4,
        result_callback=on_result)
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        # Initialize Video Capture
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Cannot open camera")
            return

        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)

            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Process the frame and find hands
            timestamp = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp)

            # Draw the hand annotations on the frame
            if latest_result and latest_result.hand_landmarks:
                annotated_image = np.copy(frame)
                for hand_landmarks in latest_result.hand_landmarks:
                    # Draw landmarks
                    for landmark in hand_landmarks:
                        x, y = int(landmark.x * annotated_image.shape[1]), int(landmark.y * annotated_image.shape[0])
                        cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

                    # Draw connections
                    if HAND_CONNECTIONS:
                        for connection in HAND_CONNECTIONS:
                            start_idx = connection[0]
                            end_idx = connection[1]
                            start_point = (int(hand_landmarks[start_idx].x * annotated_image.shape[1]),
                                           int(hand_landmarks[start_idx].y * annotated_image.shape[0]))
                            end_point = (int(hand_landmarks[end_idx].x * annotated_image.shape[1]),
                                         int(hand_landmarks[end_idx].y * annotated_image.shape[0]))
                            cv2.line(annotated_image, start_point, end_point, (255, 0, 0), 2)
                
                # Display the resulting frame
                cv2.imshow('Hand Tracking', annotated_image)
            else:
                 # Display the resulting frame
                cv2.imshow('Hand Tracking', frame)


            # Break the loop on 'q' key press
            if cv2.waitKey(1) == ord('q'):
                break

        # Release the capture and destroy all windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
