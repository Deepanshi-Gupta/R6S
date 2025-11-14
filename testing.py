import cv2
from ultralytics import YOLO
import easyocr
import datetime

# -----------------------------#
# File Paths and Configurations
# -----------------------------#
#video_path = r"C:\Users\aayus\Downloads\test_data.mp4"
video_path = r"C:\Users\aayus\Downloads\ocr+kill - Made with Clipchamp.mp4"
model1_path = r"C:\Users\aayus\Downloads\Yolo11_bomb.pt"   # Model 1: Bomb detection
model2_path = r"C:\Users\aayus\Downloads\Yolo11_cross.pt"  # Model 2: Cross detection
skip_frames = 2  # Process every 5th frame

roi_coords = (950, 180, 1320, 650)  # Region of interest for OCR

# -----------------------------#
# Load Models
# -----------------------------#
model_bomb = YOLO(model1_path)
model_cross = YOLO(model2_path)
reader = easyocr.Reader(['en'])

# -----------------------------#
# Setup Video Capture
# -----------------------------#
cap = cv2.VideoCapture(video_path)
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)

print("-" * 60)
print(f"Analysis Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 60)

# Initialize flags, counters, streak trackers
active_ocr = False
ocr_active_frames = 0
plant_defuse_ocr_detected = False

action_counts = {}

# Cross detection
cross_streak = 0
cross_kill_count = 0

# Plant defuse OCR streak tracking
plant_defuse_streak = 0
plant_defuse_count = 0
plant_defuse_active = False

# Plant deploy OCR streak tracking
plant_deploy_streak = 0
gadget_deploy_count = 0
plant_deploy_active = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % (skip_frames + 1) != 0:
        frame_count += 1
        continue

    timestamp_sec = frame_count / fps
    timestamp_str = f"{int(timestamp_sec // 60):02d}:{int(timestamp_sec % 60):02d}"

    results_bomb = model_bomb.predict(source=[frame], save=False, verbose=False)
    results_cross = model_cross.predict(source=[frame], save=False, verbose=False)

    show_frame = frame.copy()

    # Track if OCR should activate this frame based on bomb detections
    ocr_triggered_this_frame = False

    # Flags per frame
    cross_detected_this_frame = False
    plant_defuse_ocr_detected_this_frame = False
    plant_deploy_ocr_detected_this_frame = False

    # Process Cross detections to mark presence and draw boxes
    for r in results_cross:
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            cross_detected_this_frame = True
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                label = model_cross.names[int(box.cls[0])] if hasattr(model_cross, "names") else str(int(box.cls[0]))
                conf = float(box.conf[0])
                cv2.rectangle(show_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(show_frame, f"{label} {conf:.2f} @{timestamp_str}", (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(f"[{timestamp_str}][CrossDetection] {label}: {conf:.2f}")

    # Cross kill logic for consecutive frames
    if cross_detected_this_frame:
        cross_streak += 1
    else:
        if cross_streak >= 2:  # Threshold > 2 for kill count
            cross_kill_count += 1
            print(f"*** Kill count incremented to: {cross_kill_count} ***")
        cross_streak = 0

    # Process Bomb detections for labels and possible OCR activation
    current_action = None
    for r in results_bomb:
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                label = model_bomb.names[int(box.cls[0])] if hasattr(model_bomb, "names") else str(int(box.cls[0]))
                conf = float(box.conf[0])
                cv2.rectangle(show_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(show_frame, f"{label} {conf:.2f} @{timestamp_str}", (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                print(f"[{timestamp_str}][BombDetection] {label}: {conf:.2f}")

                if label.lower() == "plant defuse":
                    active_ocr = True
                    ocr_active_frames = 15
                    ocr_triggered_this_frame = True
                    current_action = None  # Do not count yet, wait for OCR confirmation
                elif label.lower() in ['deploy', 'kill']:
                    active_ocr = True
                    ocr_active_frames = 10
                    ocr_triggered_this_frame = True
                    current_action = label.lower()

    # OCR processing if active
    if active_ocr and ocr_active_frames > 0:
        x1, y1, x2, y2 = roi_coords
        h, w = show_frame.shape[:2]
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        roi_img = show_frame[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
        ocr_results = reader.readtext(roi_rgb, detail=1)

        for bbox, text, conf in ocr_results:
            text_lower = text.lower()
            pt1 = tuple(map(int, bbox[0]))
            pt2 = tuple(map(int, bbox[2]))
            cv2.rectangle(roi_img, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(roi_img, text, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            print(f"[{timestamp_str}][OCR] {text} (conf={conf:.2f})")

            # Track plant defuse OCR text streak
            if ("deluser planted" in text_lower or "defuser planted" in text_lower or "plant defuse" in text_lower):
                plant_defuse_ocr_detected_this_frame = True

            # Track plant deploy OCR text streak
            if "deploy" in text_lower:
                plant_deploy_ocr_detected_this_frame = True

            # Confirm plant defuse count during OCR session (only once per session)
            if "plant defuse" in text_lower and not plant_defuse_ocr_detected:
                action_counts["plant defuse"] = action_counts.get("plant defuse", 0) + 1
                plant_defuse_ocr_detected = True
                print(f"*** Action Count Update: plant defuse = {action_counts['plant defuse']} (confirmed by OCR) ***")

        show_frame[y1:y2, x1:x2] = roi_img

        ocr_active_frames -= 1
        if ocr_active_frames <= 0:
            active_ocr = False
            plant_defuse_ocr_detected = False  # Reset for next OCR session

    # --- Plant Defuse OCR streak counting ---
    if plant_defuse_ocr_detected_this_frame:
        plant_defuse_streak += 1
        plant_defuse_active = True
    else:
        if plant_defuse_active and plant_defuse_streak >= 4:  # Threshold 4 frames
            plant_defuse_count += 1
            print(f"*** Plant Defuse Action counted by OCR streak: {plant_defuse_count} ***")
        plant_defuse_streak = 0
        plant_defuse_active = False

    # --- Plant Deploy OCR streak counting ---
    if plant_deploy_ocr_detected_this_frame:
        plant_deploy_streak += 1
        plant_deploy_active = True
    else:
        if plant_deploy_active and plant_deploy_streak >= 4:  # Threshold 4 frames
            gadget_deploy_count += 1
            print(f"*** Plant Deploy Action counted by OCR streak: {gadget_deploy_count} ***")
        plant_deploy_streak = 0
        plant_deploy_active = False

    fixed_width, fixed_height = 640, 360
    show_frame = cv2.resize(show_frame, (fixed_width, fixed_height))
    cv2.imshow("Games", show_frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to quit
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# Add cross kills to action counts dictionary for reporting
action_counts["cross_kills"] = cross_kill_count
action_counts["plant_defuse"] = action_counts.get("plant defuse", 0) + plant_defuse_count
action_counts["plant_deploy"] = gadget_deploy_count

# After the video ends, print final counts:
print("\n============ Final Counts ============")
print(f"Total Kills (Cross): {cross_kill_count}")
print(f"Total Plant Defuse : {plant_defuse_count}")
print(f"Total Gadgets Deploy : {gadget_deploy_count}")
print("=====================================")
