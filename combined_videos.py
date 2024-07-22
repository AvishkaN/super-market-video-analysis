import cv2
import numpy as np
import os

# Open video captures
cap = cv2.VideoCapture('data/12new.mp4')
cap1 = cv2.VideoCapture('data/11new.mp4')

# Check if videos are opened successfully
if not cap.isOpened() or not cap1.isOpened():
    print("Error: One or more videos failed to open.")
    exit()

# Get properties of the first video
fps = min(cap.get(cv2.CAP_PROP_FPS), cap1.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define new dimensions for output frames (reduce size by half)
new_width = int(frame_width * 0.5)
new_height = int(frame_height * 0.5)

# Specify the output directory (create if it doesn't exist)
output_dir = 'data/output'
os.makedirs(output_dir, exist_ok=True)

# Define the output video file path
output_file = os.path.join(output_dir, 'combined_output.avi')

# Create VideoWriter object to save combined output with reduced size
out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), fps, (new_width, new_height * 2))

while True:
    ret, frame = cap.read()
    ret1, frame1 = cap1.read()

    if ret and ret1: # If both frames are read successfully
        # Resize frames to new dimensions
        frame = cv2.resize(frame, (new_width, new_height))
        frame1 = cv2.resize(frame1, (new_width, new_height))

        # Concatenate frames vertically
        combined_frame = np.concatenate((frame, frame1), axis=0)

        # Display the combined frame (optional)
        cv2.imshow('Frame', combined_frame)

        # Write the frame to the output video file
        out.write(combined_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release video captures and writer
cap.release()
cap1.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
