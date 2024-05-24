# src/capture/video_capture.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.detection.nudity_detection import detect_nudity
from src.blurring.blur import apply_blur

def capture_video():
    print("Starting video capture...")
    cap = cv2.VideoCapture(0)  # 0 for default camera

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    fig, ax = plt.subplots()
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return

    img_display = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def update_frame(i):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            return img_display,

        print("Frame captured, performing detection...")

        # Detect nudity regions
        nudity_regions = detect_nudity(frame)
        print(f"Nudity regions detected: {nudity_regions}")

        # Apply blurring to detected nudity regions
        frame = apply_blur(frame, nudity_regions)
        print("Blurring applied to detected regions.")

        img_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return img_display,

    ani = FuncAnimation(fig, update_frame, interval=50, blit=True)
    plt.show()

    cap.release()
    # Commented out due to issues
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_video()
