import cv2
import easyocr

# Video path and ROI coordinates
video_path = r"C:\Users\aayus\Downloads\ocr+kill - Made with Clipchamp.mp4"
roi_coords = (950, 180, 1320, 650)  # Define your OCR region of interest (x1, y1, x2, y2)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Open the video
cap = cv2.VideoCapture(video_path)
frame_count = 0
skip_frames = 2  # Process every 3rd frame for speed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % (skip_frames + 1) != 0:
        frame_count += 1
        continue

    # Extract ROI for OCR
    x1, y1, x2, y2 = roi_coords
    roi_img = frame[y1:y2, x1:x2]
    roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)

    # Perform OCR on ROI
    ocr_results = reader.readtext(roi_rgb)
    for bbox, text, conf in ocr_results:
        pt1 = tuple(map(int, bbox[0]))
        pt2 = tuple(map(int, bbox[2]))
        cv2.rectangle(roi_img, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(roi_img, text, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        print(f"Frame {frame_count} OCR Text: {text} (Confidence: {conf:.2f})")

    # Show the ROI with OCR annotations
    cv2.imshow('OCR Region', roi_img)

    # Break on ESC key
    if cv2.waitKey(1) == 27:
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
