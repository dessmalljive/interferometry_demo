import cv2 as cv
import numpy as np

# Code from Google AI

def detect_magnets_robust(frame):
    # 1. Convert to grayscale and apply a heavy blur to smooth marker lines
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7, 7), 0)
    
    # 2. Adaptive thresholding handles uneven lighting across the whiteboard
    thresh = cv.adaptiveThreshold(
        blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv.THRESH_BINARY_INV, 21, 10
    )
    
    # 3. Morphological cleaning: removes stray pen marks or tiny specks
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    
    # 4. Find all outlines (contours)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        # Filter by size: Ignore tiny noise and massive objects
        area = cv.contourArea(cnt)
        if not (100 < area < 400):  # Adjust based on your camera distance
            continue
            
        # Filter by Aspect Ratio (Is it roughly square/round or a long line?)
        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if not (0.75 <= aspect_ratio <= 1.35):
            continue
            
        # Filter by Circularity / Compactness: (4 * pi * Area) / (Perimeter^2)
        # A perfect circle is 1.0. High-contrast shapes or text will be very low.
        perimeter = cv.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        
        if circularity > 0.65:  # Closer to 1.0 means more circular
            # Success! Draw the bounding circle around the magnet
            (cx, cy), radius = cv.minEnclosingCircle(cnt)
            center = (int(cx), int(cy))
            radius = int(radius)
            
            cv.circle(frame, center, radius, (0, 255, 0), 2)
            cv.circle(frame, center, 2, (0, 0, 255), -1)
            
    return frame

def main():
    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed = detect_magnets_robust(frame)
        cv.imshow('Robust Magnet Tracker', processed)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
