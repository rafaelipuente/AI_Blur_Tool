# src/blurring/blur.py

import cv2

def apply_blur(frame, regions):
    """
    Apply blurring effect to the specified regions in the video frame.

    Parameters:
    - frame: The video frame to be processed.
    - regions: List of bounding box coordinates for the regions to blur.

    Returns:
    - frame: The processed video frame with blurred regions.
    """
    for (x, y, w, h) in regions:
        # Ensure that width and height are positive
        if w > 0 and h > 0:
            # Extract the region of interest (ROI) from the frame
            roi = frame[y:y+h, x:x+w]
            # Apply Gaussian blur to the ROI
            blurred_roi = cv2.GaussianBlur(roi, (23, 23), 30)
            # Replace the original ROI with the blurred ROI
            frame[y:y+h, x:x+w] = blurred_roi
        else:
            print(f"Skipping invalid region: {(x, y, w, h)}")

    return frame
