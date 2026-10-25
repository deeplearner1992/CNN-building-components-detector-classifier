import cv2
from ultralytics import YOLO
import time

# 1. Load your YOLO model
# model = YOLO("yolov8n.pt")  # replace with your custom model if needed
model = YOLO("runs/detect/train/weights/best.pt")
class_names = model.names

# 2. Initialize VideoCapture (0 for built-in camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

last_trigger_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # 3. Perform inference on the current frame
    # stream=True uses a generator which is highly recommended for real-time video processing
    results = model(frame, imgsz=640, conf=0.6, stream=True)

    # 4. Extract and draw the box data
    for result in results:
        boxes = result.boxes.data
        for box in boxes:
            # print(box)
            # break
            # Get integer coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2, conf, lbl = box.tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get confidence and class metadata
            confidence = float(conf)
            class_id = int(lbl)
            label = class_names[lbl]
            
            # Format text overlay
            text = f"{label} {confidence:.2f}"
            
            # Draw bounding box (Green)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw background rectangle for label readability
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 0), -1)
            
            # Put label text (White text over green background)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 5. Show the real-time stream
    cv2.imshow("YOLO Real-Time Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# 6. Clean up resources
cap.release()
cv2.destroyAllWindows()
