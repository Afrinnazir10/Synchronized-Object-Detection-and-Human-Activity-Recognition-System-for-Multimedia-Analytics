import os
import cv2

def convert_videos_to_frames(input_path, output_path):
    # Get the list of class directories
    class_directories = os.listdir(input_path)

    for class_dir in class_directories:
        class_input_dir = os.path.join(input_path, class_dir)
        class_output_dir = os.path.join(output_path, class_dir)

        # Create output directory for class if it doesn't exist
        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)

        # Get list of video files in class directory
        video_files = [f for f in os.listdir(class_input_dir) if f.endswith('.mp4')]

        for video_file in video_files:
            video_path = os.path.join(class_input_dir, video_file)

            # Open video file
            video_capture = cv2.VideoCapture(video_path)
            success, image = video_capture.read()
            count = 0

            # Read frames from video and save as images
            while success:
                frame_output_path = os.path.join(class_output_dir, f"{video_file}_{count}.jpg")
                cv2.imwrite(frame_output_path, image)
                success, image = video_capture.read()
                count += 1

            video_capture.release()

# Example usage
input_path = "C:/Users/HP/Downloads/HAR Dataset/HAR VIDEOS"
output_path = "C:/Users/HP/Downloads/HAR Dataset/HAR FRAMES"

convert_videos_to_frames(input_path, output_path)
