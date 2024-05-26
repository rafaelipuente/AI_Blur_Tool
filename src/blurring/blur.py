# src/blurring/blur.py

import cv2

def apply_blur(frame, regions):
    for (x, y, w, h) in regions:
        if w > 0 and h > 0:  # Ensure width and height are positive
            roi = frame[y:y+h, x:x+w]
            blurred_roi = cv2.GaussianBlur(roi, (23, 23), 30)
            frame[y:y+h, x:x+w] = blurred_roi
    return frame
