import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
from collections import deque
from ultralytics import YOLO

# Load YOLO model for object detection
object_detection_model = YOLO("C:/Users/HP/PycharmProjects/OD/model_weights/yolov8x.pt")

# Load LSTM model for human activity recognition
activity_recognition_model = load_model('HAR3333.h5')

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Define labels for objects and activities
object_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bow', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

activity_labels = ['Carrying', 'Checking the time on a watch', 'Drinking', 'Eating', 'Reading', 'Standing', 'Talking through phone', 'Using a computer', 'Walking', 'Writing']

# Initialize deque to store recent frames
buffer_size = 33
frame_buffer = deque(maxlen=buffer_size)

# Open the video file
video_file = 'C:/Users/HP/Downloads/HAR Dataset/HAR VIDEOS/Drinking/3.mp4'  # Replace with the path to your video file
cap = cv2.VideoCapture(video_file)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create an output video writer
output_file = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use MP4 codec
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame to detect poses
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(frame_rgb)

    if results_pose.pose_landmarks is not None:
        # Extract pose landmarks
        pose_landmarks = [[lm.x, lm.y, lm.z] for lm in results_pose.pose_landmarks.landmark]
        frame_buffer.append(pose_landmarks)

        # If the window is filled, predict activity
        if len(frame_buffer) == buffer_size:
            # Preprocess the frames
            processed_frames = np.array(frame_buffer)  # Convert to numpy array
            processed_frames = processed_frames.reshape(-1, 33, 3)  # Reshape to match model input shape

            # Predict activity
            predictions_activity = activity_recognition_model.predict(processed_frames)
            average_confidence = np.mean(predictions_activity, axis=0)

            # Get predicted activity
            predicted_activity = activity_labels[np.argmax(average_confidence)]
            confidence = average_confidence[np.argmax(average_confidence)]
            print(f"Predicted Activity: {predicted_activity} (Confidence: {confidence:.2f})")  # Debug print

            # Object detection
            results_object_list = object_detection_model.predict(frame, show=False)

            # Extract detected objects and their confidence scores
            detected_objects = []
            for results_object in results_object_list:
                if len(results_object.xyxy) > 0:
                    detected_objects.extend([object_labels[int(obj[-1])] for obj in results_object.xyxy[0]])
                else:
                    print("No objects detected.")  # Debug print

            # Form sentences
            sentences = [f"{obj} {predicted_activity}" for obj in detected_objects]

            # Combine sentences
            combined_sentence = ', '.join(sentences)

            # Display combined sentence on the frame
            cv2.putText(frame, combined_sentence, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)

    # Display the frame
    cv2.imshow('ODHAR', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer, and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()


