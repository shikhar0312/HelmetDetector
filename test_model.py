from ultralytics import YOLO

model = YOLO("models/helmet.pt")

results = model("videos/photo.jpg", show=True, save=True)
