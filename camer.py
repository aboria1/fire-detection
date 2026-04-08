import cv2
from ultralytics import YOLO

# 1. Load your trained model
# Replace 'path/to/your/best.pt' with the actual path to your weights file
# (Usually found in runs/detect/train/weights/best.pt after training)
model = YOLO('best.pt') 

# 2. Open the laptop camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting fire detection. Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Run YOLOv8 inference on the frame
    # stream=True is more efficient for real-time video
    # results = model(frame, stream=True)
    results = model(frame, conf=0.5, stream=True)

    # 4. Process results and visualize
    for r in results:
        # This draws the bounding boxes and labels on the frame
        annotated_frame = r.plot()

    # Display the resulting frame
    cv2.imshow('Fire & Smoke Detection', annotated_frame)

    # 5. Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()