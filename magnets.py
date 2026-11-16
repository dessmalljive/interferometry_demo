import cv2
import numpy as np

# Code from Google AI

def detect_magnets_robust(frame):
    # 1. Convert to grayscale and apply a heavy blur to smooth marker lines
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 2. Adaptive thresholding handles uneven lighting across the whiteboard
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 10
    )
    
    # 3. Morphological cleaning: removes stray pen marks or tiny specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 4. Find all outlines (contours)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        # Filter by size: Ignore tiny noise and massive objects
        area = cv2.contourArea(cnt)
        if not (150 < area < 8000):  # Adjust based on your camera distance
            continue
            
        # Filter by Aspect Ratio (Is it roughly square/round or a long line?)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if not (0.75 <= aspect_ratio <= 1.35):
            continue
            
        # Filter by Circularity / Compactness: (4 * pi * Area) / (Perimeter^2)
        # A perfect circle is 1.0. High-contrast shapes or text will be very low.
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        
        if circularity > 0.65:  # Closer to 1.0 means more circular
            # Success! Draw the bounding circle around the magnet
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            center = (int(cx), int(cy))
            radius = int(radius)
            
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.circle(frame, center, 2, (0, 0, 255), -1)
            
    return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed = detect_magnets_robust(frame)
        cv2.imshow('Robust Magnet Tracker', processed)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
