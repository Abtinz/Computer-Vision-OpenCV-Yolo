from ultralytics import YOLO

model = YOLO("yolov10n.pt")
print("yoloV10 before training on coco8",model)

model.train(data="coco8.yaml", epochs=100, imgsz=640)
print("yoloV10 after training on coco8",model)

results = model('Yolo\yolo10\coco_trained\image.jpg')
print(results)
results[0].show()

results2 = model('Yolo\yolo10\coco_trained\image2.jpg')
print(results2)
results2[0].show()
