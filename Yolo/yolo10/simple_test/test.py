from ultralytics import YOLO
import cv2

model = YOLO("yolov10n.pt")
print(model)

results = model('Yolo\yolo10\image.jpg')
print(results)
results[0].show()

results2 = model('Yolo\yolo10\image2.jpg')
print(results2)
results2[0].show()
