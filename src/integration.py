import cv2
import numpy as np
import sys
from pathlib import Path

from keras.models import load_model
import mediapipe as mp
from collections import deque
from ultralytics import YOLO

# ============================================================
# 1. LOAD MODELS
# ============================================================

print("Loading YOLO model...")
object_detection_model = YOLO("model_weights/yolov8x.pt")

print("Loading HAR model...")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

activity_recognition_model = load_model(
    PROJECT_ROOT / "models" / "HAR3333.h5"
)

print("Both models loaded successfully!")


# ============================================================
# 2. ACTIVITY LABELS
# ============================================================

activity_labels = [
    "Carrying",
    "Checking the time on a watch",
    "Drinking",
    "Eating",
    "Reading",
    "Standing",
    "Talking through phone",
    "Using a computer",
    "Walking",
    "Writing"
]


# ============================================================
# 3. MEDIAPIPE POSE
# ============================================================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5
)


# ============================================================
# 4. VIDEO
# ============================================================

if len(sys.argv) < 2:
    print("Usage: python tests/test_integration.py <video_path>")
    sys.exit(1)

video_file = Path(sys.argv[1])

if not video_file.is_file():
    print(f"ERROR: Video file not found: {video_file}")
    sys.exit(1)

cap = cv2.VideoCapture(str(video_file))

if not cap.isOpened():
    print(f"ERROR: Could not open video: {video_file}")
    sys.exit(1)

print("Video opened successfully!")


# ============================================================
# 5. FRAME BUFFER
# ============================================================

buffer_size = 33

frame_buffer = deque(
    maxlen=buffer_size
)


# ============================================================
# 6. PROCESS VIDEO
# ============================================================

frame_number = 0
activity_predictions = 0

max_frames = None

while max_frames is None or frame_number < max_frames:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------------------
    # MediaPipe Pose
    # --------------------------------------------------------

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results_pose = pose.process(
        frame_rgb
    )

    predicted_activity = "Waiting..."
    activity_confidence = 0.0


    # --------------------------------------------------------
    # HAR
    # --------------------------------------------------------

    if results_pose.pose_landmarks is not None:

        pose_landmarks = [
            [lm.x, lm.y, lm.z]
            for lm in results_pose.pose_landmarks.landmark
        ]

        frame_buffer.append(
            pose_landmarks
        )

        if len(frame_buffer) == buffer_size:

            processed_frames = np.array(
                frame_buffer
            )

            processed_frames = processed_frames.reshape(
                -1,
                33,
                3
            )

            predictions = activity_recognition_model.predict(
                processed_frames,
                verbose=0
            )

            average_confidence = np.mean(
                predictions,
                axis=0
            )

            predicted_index = np.argmax(
                average_confidence
            )

            predicted_activity = activity_labels[
                predicted_index
            ]

            activity_confidence = average_confidence[
                predicted_index
            ]

            activity_predictions += 1


    # --------------------------------------------------------
    # YOLO Object Detection
    # --------------------------------------------------------

    results_object = object_detection_model.predict(
        frame,
        verbose=False
    )

    result = results_object[0]

    detected_objects = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            object_name = object_detection_model.names[
                class_id
            ]

            # Keep reasonably confident detections
            if confidence >= 0.50:

                detected_objects.append(
                    object_name
                )


    # Remove duplicate object names
    detected_objects = list(
        dict.fromkeys(detected_objects)
    )


    # --------------------------------------------------------
    # Combined result
    # --------------------------------------------------------

    if detected_objects:

        combined_sentence = ", ".join(
            f"{obj} {predicted_activity}"
            for obj in detected_objects
        )

    else:

        combined_sentence = (
            f"No objects | {predicted_activity}"
        )


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Activity: {predicted_activity}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {activity_confidence:.2f}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Objects: {', '.join(detected_objects)}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "OD + HAR Integration Test",
        frame
    )


    # --------------------------------------------------------
    # Print combined result
    # --------------------------------------------------------

    if frame_number % 10 == 0:

        print(
            f"Frame {frame_number} | "
            f"Activity: {predicted_activity} | "
            f"Objects: {detected_objects}"
        )


    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# 7. CLEANUP
# ============================================================

cap.release()
pose.close()
cv2.destroyAllWindows()


print()
print("========================================")
print("INTEGRATION TEST COMPLETED")
print("========================================")
print("Frames processed:", frame_number)
print("HAR predictions:", activity_predictions)
