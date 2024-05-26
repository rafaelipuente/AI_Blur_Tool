import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import cv2
import numpy as np
import tensorflow as tf
from src.detection.nudity_detection import detect_nudity
from src.blurring.blur import Blurring

def capture_video():
    cap = cv2.VideoCapture(0)  # Change to appropriate video source
    blurring = Blurring()
    frame_buffer = []

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect nudity regions
            nudity_regions = detect_nudity(frame)
            print(f"Detected nudity regions: {nudity_regions}")

            # Apply blur to nudity regions
            blurred_frame = blurring.apply_blur(frame, nudity_regions)
            frame_buffer.append(blurred_frame)

            # Write the frame into the file 'output.avi'
            out.write(blurred_frame)

            # Display the frame
            cv2.imshow('Blurred Frame', blurred_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Video capture interrupted")
    finally:
        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Save frame buffer to file
        np.save("saved_model/frames_array.npy", np.array(frame_buffer))
        print("Frame buffer saved to 'saved_model/frames_array.npy'")

if __name__ == "__main__":
    capture_video()
