import cv2
import numpy as np

class Blurring:
    def __init__(self):
        self.previous_regions = []

    def apply_circular_blur(self, frame, center, radius):
        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.circle(mask, center, radius, (255, 255, 255), -1)
        blurred_image = cv2.GaussianBlur(frame, (23, 23), 30)
        result = np.where(mask == (255, 255, 255), blurred_image, frame)
        return result

    def apply_blur(self, frame, regions):
        current_regions = []
        for (x, y, w, h) in regions:
            if w > 0 and h > 0:
                center = (x + w // 2, y + h // 2)
                radius = max(w, h) // 2
                current_regions.append((center, radius))
        
        # Check if the regions are consistent with previous detections
        consistent_regions = self.temporal_consistency_check(current_regions)

        for (center, radius) in consistent_regions:
            frame = self.apply_circular_blur(frame, center, radius)

        self.previous_regions = current_regions
        return frame

    def temporal_consistency_check(self, current_regions):
        if not self.previous_regions:
            return current_regions
        
        consistent_regions = []
        for curr_center, curr_radius in current_regions:
            for prev_center, prev_radius in self.previous_regions:
                distance = np.linalg.norm(np.array(curr_center) - np.array(prev_center))
                if distance < (curr_radius + prev_radius) // 2:
                    consistent_regions.append((curr_center, curr_radius))
                    break
        return consistent_regions
