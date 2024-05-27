import sys
import os
import cv2
import numpy as np
from threading import Event
from src.detection.nudity_detection import detect_nudity
from src.blurring.blur import Blurring

def capture_video(video_source, output_file, stop_event):
    try:
        video_source = int(video_source)
    except ValueError:
        pass
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    blurring = Blurring()
    frame_buffer = []

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (640,480))

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            nudity_regions = detect_nudity(frame)
            blurred_frame = blurring.apply_blur(frame, nudity_regions)
            frame_buffer.append(blurred_frame)

            out.write(blurred_frame)
            cv2.imshow('Blurred Frame', blurred_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Video capture interrupted")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        np.save("saved_model/frames_array.npy", np.array(frame_buffer))
        print("Frame buffer saved to 'saved_model/frames_array.npy'")

if __name__ == "__main__":
    capture_video(0, 'output.avi', Event())
