import cv2
import numpy as np
import math

# Function to detect lanes
def detect_lanes(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Define region of interest
    mask = np.zeros_like(edges)
    height, width = frame.shape[:2]
    vertices = np.array([[(0, height), (width/2, height/2), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Apply Hough Transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 20, np.array([]), minLineLength=30, maxLineGap=100)
    
    return lines

# Function to draw lanes on the frame
def draw_lanes(frame, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

# Function for lane keep assist
def lane_keep_assist(frame):
    lines = detect_lanes(frame)
    draw_lanes(frame, lines)
    # Implement your steering control logic here
    # For demonstration, let's just print the detected lines
    print("Detected lines:", lines)

# Main function
def main():
    cap = cv2.VideoCapture(0)  # You can change the parameter to read from a video file
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            lane_keep_assist(frame)
            cv2.imshow('Lane Keep Assist', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()