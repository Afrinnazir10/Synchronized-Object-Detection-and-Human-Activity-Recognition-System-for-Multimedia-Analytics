import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
from collections import deque

# Function to preprocess the frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (33, 3))  # Resize frame to match model input shape
    frame = frame / 255.0  # Normalize pixel values
    return frame

# Load the pre-trained LSTM model
model = load_model('HAR3333.h5')

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Define labels for activities
labels = ['Carrying', 'Checking the time on a watch', 'Drinking', 'Eating', 'Reading', 'Standing', 'Talking through phone', 'Using a computer', 'Walking', 'Writing']  # Replace with your actual action labels

# Initialize deque to store recent frames
buffer_size = 33
frame_buffer = deque(maxlen=buffer_size)

# Open the video file
video_file = 'C:/Users/HP/Downloads/HAR Dataset/HAR VIDEOS/Reading/3.mp4'  # Replace with the path to your video file
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
    results = pose.process(frame_rgb)

    if results.pose_landmarks is not None:
        # Extract pose landmarks
        pose_landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
        frame_buffer.append(pose_landmarks)

        # If the window is filled, predict activity
        if len(frame_buffer) == buffer_size:
            # Preprocess the frames
            processed_frames = np.array(frame_buffer)  # Convert to numpy array
            processed_frames = processed_frames.reshape(-1, 33, 3)  # Reshape to match model input shape

            # Predict activity
            predictions = model.predict(processed_frames)
            average_confidence = np.mean(predictions, axis=0)

            # Get predicted activity
            predicted_activity = labels[np.argmax(average_confidence)]
            confidence = average_confidence[np.argmax(average_confidence)]

            # Print predicted activity and confidence
            print(f"Predicted Activity: {predicted_activity} (Confidence: {confidence:.2f})")

            # Display predicted activity on the frame
            cv2.putText(frame, f"Activity: {predicted_activity} ({confidence:.2f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)

    # Display the frame
    cv2.imshow('Activity Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer, and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()
