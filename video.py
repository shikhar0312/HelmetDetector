from ultralytics import YOLO
import cv2
import os

# Load model
model = YOLO("models/helmet.pt")

# Load video
cap = cv2.VideoCapture("videos/test.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create outputs folder if not exists
os.makedirs("outputs", exist_ok=True)

# Create video writer
out = cv2.VideoWriter(
    "outputs/output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run detection
    results = model(frame)
    annotated = results[0].plot()

    # Add timestamp
    video_time = frame_count / fps
    timestamp = f"Time: {video_time:.2f}s"

    cv2.putText(
        annotated,
        timestamp,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Save frame
    out.write(annotated)

    # Show frame
    cv2.imshow("Helmet Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved to outputs/output.mp4")
