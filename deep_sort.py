import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Config values
video_path = "D:/HUST/dev/py/RAT_DETECTION/data_chua_label/test/Night Vision Rat Hunt.mp4"
conf_threshold = 0.3  # Lowered for testing
tracking_class = None  # Track all classes for testing

# Initialize DeepSORT
tracker = DeepSort(max_iou_distance=0.7,
                    max_age=30,
                    n_init=3)

# Initialize YOLOv8
device = "cpu"  # Use CPU for compatibility, change to "cuda" or "mps:0" if needed
model = YOLO("D:/HUST/dev/py/RAT_DETECTION/model/best_type02_topANDfront_05.pt")

# Load class names from file
with open("D:/HUST/dev/py/RAT_DETECTION/model/class.names") as f:
    class_names = f.read().strip().split('\n')

colors = np.random.randint(0, 255, size=(len(class_names), 3))

# Initialize VideoCapture
cap = cv2.VideoCapture(video_path)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    frame_count += 1
    print(f"Processing frame {frame_count}")

    # Run YOLOv8 detection
    results = model(frame)

    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        print(f"YOLOv8 detections: {len(boxes)}")

        for box, cls, conf in zip(boxes, classes, confidences):
            if conf < conf_threshold:
                continue
            if tracking_class is None or int(cls) == tracking_class:
                x1, y1, x2, y2 = map(int, box)
                detections.append([[x1, y1, x2-x1, y2-y1], conf, int(cls)])
                print(f"Detection: Class {int(cls)}, Confidence {conf:.2f}, Box {[x1, y1, x2, y2]}")

    # Update tracks using DeepSORT
    tracks = tracker.update_tracks(detections, frame=frame)
    print(f"DeepSORT tracks: {len(tracks)}")

    # Draw bounding boxes and labels
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        class_id = track.get_det_class()
        x1, y1, x2, y2 = map(int, ltrb)
        color = colors[class_id]
        label = f"{class_names[class_id]}-{track_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color.tolist(), 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(label) * 12, y1), color.tolist(), -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        print(f"Drew box for track {track_id}: Class {class_id}, Box {[x1, y1, x2, y2]}")

    # Display the frame
    cv2.imshow("YOLOv8 + DeepSORT", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()