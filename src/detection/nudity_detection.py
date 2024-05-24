# src/detection/nudity_detection.py

from nudenet import NudeDetector
import cv2
import tempfile
import os

# Initialize the NudeDetector
detector = NudeDetector()

def detect_nudity(frame):
    """
    Detect nudity in the given video frame using NudeNet.

    Parameters:
    - frame: The video frame in which to detect nudity.

    Returns:
    - regions: List of bounding box coordinates for detected nudity regions.
    """
    # Convert the frame to RGB format expected by NudeNet
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Use a temporary file to save the frame
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, 'temp_frame.jpg')
    
    cv2.imwrite(temp_file_path, frame_rgb)
    
    # Perform nudity detection
    results = detector.detect(temp_file_path)
    
    regions = []
    height, width, _ = frame.shape

    for result in results:
        x, y, w, h = result['box']
        # Ensure width and height are positive
        w = abs(w)
        h = abs(h)
        
        # Filter out detections based on typical face region size and location
        # This is a heuristic and might need adjustment based on your specific use case
        if y > height / 3 and w > width / 10 and h > height / 10:
            regions.append((int(x), int(y), int(w), int(h)))
        else:
            print(f"Skipping non-private region: {(x, y, w, h)}")
    
    # Clean up temporary file
    os.remove(temp_file_path)
    
    return regions
