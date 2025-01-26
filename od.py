from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import cv2

model = YOLO("C:/Users/HP/PycharmProjects/OD/model_weights/yolov8x.pt")
results = model.predict(source="0" , show=True)

print(results)
