from picamera2 import Picamera2
import time

# Initialize the Picamera2 object
picam2 = Picamera2()

# Start the camera preview
picam2.start_preview()

# Give the camera a moment to adjust settings
time.sleep(2)

# Set the camera to record video
picam2.start_recording("/home/pi/video.h264")

print("Recording video... Press Ctrl+C to stop.")

try:
    # Record for 10 seconds
    time.sleep(10)
finally:
    # Stop recording and release the camera
    picam2.stop_recording()
    picam2.close()
    print("Recording stopped.")

# End of script
