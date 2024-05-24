# src/detection/object_detection.py
#libraries
import cv2
import numpy as np

# Load YOLO model
# Load the YOLO network using the configuration and weights files
net = cv2.dnn.readNet("models/yolov3.weights", "models/yolov3.cfg")

# Load the COCO class labels the YOLO model was trained on
with open("models/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get the output layer names for the YOLO model
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def detect_objects(frame):
    """
    Detect objects in the given video frame using YOLO.

    Parameters:
    - frame: The video frame in which to detect objects.

    Returns:
    - frame: The video frame with detected objects highlighted.
    """
    # Get the dimensions of the frame
    height, width, channels = frame.shape

    # Prepare the frame to be input to the YOLO model
    # Convert the frame to a blob
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Perform a forward pass of the YOLO model
    outs = net.forward(output_layers)

    # Initialize lists to hold detection information
    class_ids = []
    confidences = []
    boxes = []

    # Process each detection
    for out in outs:
        for detection in out:
            # Get the confidence score for the detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Only consider detections with confidence > 0.5
            if confidence > 0.5:
                # Calculate the coordinates of the bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Save the bounding box coordinates, confidence score, and class ID
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to remove overlapping bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the frame
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame
