import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import numpy


# Initialize empty lists to store precision, recall, and F1-score
precision_scores = []
recall_scores = []
f1_scores = []

# Process video to detect poses, objects, and recognize activities
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

            # Object detection
            results_object = object_detection_model.predict(frame, show=False)

            # Compute precision, recall, and F1-score
            precision, recall, f1, _ = precision_recall_fscore_support([1], [1], average='binary')  # Example, you need to replace [1] with actual labels
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Aggregate precision, recall, and F1-score over all frames
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f1 = np.mean(f1_scores)

# Plotting
metrics = ['Precision', 'Recall', 'F1-score']
scores = [avg_precision, avg_recall, avg_f1]

plt.figure(figsize=(8, 6))
plt.plot(metrics, scores, marker='o')
plt.title('Average Precision, Recall, and F1-score')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.grid(True)
plt.show()
