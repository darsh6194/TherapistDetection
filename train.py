from ultralytics import YOLO

if __name__ == '__main__':
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')  # You can use yolov8n, yolov8s, yolov8m, etc.

    # Train the model on your custom dataset
    results = model.train(data='custom_data.yaml', epochs=200, imgsz=640)
