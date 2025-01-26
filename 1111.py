from ultralytics import YOLO
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import tensorflow as tf
import cv2

# Load LSTM model
def load_lstm_model():
    model = tf.keras.models.load_model('HAR3333.h5')
    return model

# Load ChatGPT model
def load_chatgpt_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

def infer_lstm_model(lstm_model, frame):
    resized_frame = cv2.resize(frame, (33, 3))
    grayscale_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    lstm_input = np.expand_dims(grayscale_frame, axis=0)  # Add batch dimension
    lstm_input = np.transpose(lstm_input, (0, 2, 1))  # Transpose axes 1 and 2
    lstm_output = lstm_model.predict(lstm_input)  # Perform inference
    return lstm_output

def generate_description(chatgpt_model, tokenizer, activity_recognized, detected_objects):
    description = f"The person is {activity_recognized} with a {', '.join(detected_objects)}."
    input_ids = tokenizer.encode(description, return_tensors="pt")
    output = chatgpt_model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, max_new_tokens=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)


# Load models
lstm_model = load_lstm_model()
tokenizer, chatgpt_model = load_chatgpt_model()

# Load YOLO model
model = YOLO("C:/Users/HP/PycharmProjects/OD/model_weights/yolov8x.pt")

# Open video file
video_capture = cv2.VideoCapture('C:/Users/HP/Downloads/HAR Dataset/HAR VIDEOS/Drinking/3.mp4')  # Replace 'path_to_your_video.mp4' with your video file path

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # Perform object detection with YOLO
    results = model.predict(frame, show=False)

    # Process each result
    for result in results:
        # Extract labels of detected objects
        detected_objects = [model.names[i] for i in result.names]

        # Process LSTM output (activity recognition)
        lstm_output = infer_lstm_model(lstm_model, frame)
        activity_recognized_index = np.argmax(lstm_output)
        activity_labels = ['Carrying', 'Checking the time on a watch', 'Drinking', 'Eating', 'Reading', 'Standing', 'Talking through phone', 'Using a computer', 'Walking', 'Writing']  # Replace with your actual action labels
        activity_recognized = activity_labels[activity_recognized_index] if 0 <= activity_recognized_index < len(activity_labels) else "Unknown"

        # Generate description using ChatGPT
        full_description = generate_description(chatgpt_model, tokenizer, activity_recognized, detected_objects)

        # Display or use the output
        print(full_description)

        # Display frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
