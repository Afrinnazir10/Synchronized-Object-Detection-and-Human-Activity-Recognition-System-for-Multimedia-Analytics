import os
import cv2
import numpy as np
import mediapipe as mp

# Function to extract pose landmarks from a frame
def extract_pose_landmarks(frame, pose_model):
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect poses
    results = pose_model.process(frame_rgb)

    # Check if any poses are detected
    if results.pose_landmarks is not None:
        # Extract and return the pose landmarks
        return [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
    else:
        return None

# Function to convert frames to skeleton data
def frames_to_skeleton(frames_dir, output_dir):
    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose_model = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    # Process each frame in the frames directory
    for root, dirs, files in os.walk(frames_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):  # Assuming images are in jpg or png format
                image_path = os.path.join(root, file)

                # Load the frame
                frame = cv2.imread(image_path)

                # Extract pose landmarks from the frame
                pose_landmarks = extract_pose_landmarks(frame, pose_model)

                if pose_landmarks is not None:
                    # Create directory with label name if it doesn't exist
                    label = os.path.basename(os.path.dirname(image_path))
                    label_output_path = os.path.join(output_dir, label)
                    os.makedirs(label_output_path, exist_ok=True)

                    # Save pose landmarks as numpy array
                    skeleton_data_path = os.path.join(label_output_path, os.path.basename(image_path) + '.npy')
                    np.save(skeleton_data_path, pose_landmarks)

                    print(f"Skeleton data saved for {image_path} as {skeleton_data_path}")

    # Release resources
    pose_model.close()

# Set paths
input_frames_dir = 'C:/Users/HP/Downloads/HAR Dataset/HAR FRAMES'  # Directory containing preprocessed frames
output_skeletons_dir = 'C:/Users/HP/Downloads/HAR Dataset/HAR SKELETONS'  # Directory to save skeleton data

# Convert frames to skeleton data
frames_to_skeleton(input_frames_dir, output_skeletons_dir)
