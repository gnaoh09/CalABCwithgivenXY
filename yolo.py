from ultralytics import YOLO
import cv2

model_path = "D:/HUST/dev/py/RAT_DETECTION/model/best_type02_topANDfront_06.pt"
video_path = "D:/HUST/dev/py/RAT_DETECTION/data_chua_label/test/The New Killgerm Motion Sensor Camera - Night Time Rodent Feeding.mp4"

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)
#cap = cv2.VideoCapture(0)
thresh_hold = 0.5

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv10 inference on the frame
        results = model(frame)
        
        # Get the boxes and confidences
        boxes = results[0].boxes
        confidences = boxes.conf
        
        # Filter boxes with confidence scores >= threshold
        for box, conf in zip(boxes, confidences):
            if conf >= thresh_hold:
                # Extract coordinates and dimensions
                """ x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Display the confidence score
                label = f"{conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2) """

                frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
