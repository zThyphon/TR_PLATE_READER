from ultralytics import YOLO
from preprocessing import remove_brands_around_plate
from utils import get_device
from config import config

# get YOLO model path from config
MODEL_PATH = config["MODEL_PATH"]

# determine device
device = get_device()

print(f"[INFO] Using device: {device}")

print("[INFO] Loading YOLO model...")
detector = YOLO(MODEL_PATH)
detector.to(device)

# detect license plate with using YOLO and return detected plate
def detect_license_plate(img):
    results = detector.predict(img, device=device, conf=0.25, verbose=False)[0]

    if not results.boxes:
        return None, None

    box = results.boxes.data[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box[:4]

    cleaned_img = remove_brands_around_plate(img, x1, y1, x2, y2)
    plate = cleaned_img[y1:y2, x1:x2]

    return plate if plate.size > 0 else None, (x1, y1, x2, y2)