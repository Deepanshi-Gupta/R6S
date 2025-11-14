import cv2
from ultralytics import YOLO
import datetime

video_path = r"C:\Users\aayus\Downloads\R6S data\Gadget.mp4"
model_path = r"C:\Users\aayus\Downloads\R6S data\gadgets_specific.pt"
skip_frames = 4  # Process every 5th frame

# Initialize YOLO model
model = YOLO(model_path)

cap = cv2.VideoCapture(video_path)
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)

print("-" * 50)
print(f"Analysis Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % (skip_frames + 1) != 0:
        frame_count += 1
        continue

    timestamp_sec = frame_count / fps
    timestamp_str = f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}"

    # Run the model on the frame
    results = model.predict(source=[frame], save=False, verbose=False)

    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id] if hasattr(model, "names") else str(class_id)
                conf = float(box.conf[0])
                # Get bounding box coordinates
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Prepare label text
                label_text = f"{label}: {conf:.2f}"
                # Calculate label size
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                # Draw filled rectangle for label background
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                # Put label text above the box
                cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                print(f"[DETECTION @ {timestamp_str}] {label}: {conf:.2f}")

    # Show the frame with detections
    cv2.imshow("Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
