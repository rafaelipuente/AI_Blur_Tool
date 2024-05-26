import cv2
import torch
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('models/nudity_model/best.pt')  # Adjust the path if needed

def detect_nudity(frame):
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform the detection
    results = model(frame_rgb)

    # Extract nudity regions (assuming results format provides bounding boxes)
    nudity_regions = []
    class_names = model.names

    face_classes = [idx for idx, name in class_names.items() if 'FACE' in name]
    print(f"Face classes: {face_classes}")  # Debug print

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = class_names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w = x2 - x1
            h = y2 - y1
            if cls_id not in face_classes:  # Skip faces
                nudity_regions.append((x1, y1, w, h))
                print(f"Nudity detected: Class ID: {cls_id}, Class name: {cls_name}, Region: {(x1, y1, w, h)}")  # Detailed log
            else:
                print(f"Face detected: Class ID: {cls_id}, Class name: {cls_name}, Region: {(x1, y1, w, h)}")  # Detailed log for faces

    print(f"Final nudity regions: {nudity_regions}")  # Debug print for final nudity regions
    return nudity_regions
