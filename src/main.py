import json
import os
from PIL import Image
import cv2
import torch
from ultralytics import YOLO
from cnn.cnn_model import SerbianBanknoteCNN
from torchvision import transforms

# -------------------------------
# Configuration
# -------------------------------
with open("./configs/cnn_config.json", "r") as f:
    config = json.load(f)

CNN_WEIGHTS = config['model']['weights_file']
SAVE_DIR = config['model']['save_dir']
IMG_SIZE  = config['model']['img_size']
CLASS_NAMES = config['classes']
NUM_CLASSES = len(CLASS_NAMES)
YOLO_WEIGHTS = r"..\runs\train\serbian_banknotes\weights\best.pt"

# -------------------------------
# YOLO load
# -------------------------------
yolo_model = YOLO(YOLO_WEIGHTS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = SerbianBanknoteCNN(NUM_CLASSES).to(device)
cnn_model.load_state_dict(torch.load(SAVE_DIR+'/'+CNN_WEIGHTS, map_location=device))
cnn_model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# -------------------------------
# Enter image path
# -------------------------------
image_path = input("Enter the absolute path to the image ('exit'): ").strip()
while image_path != 'exit':
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")


    # -------------------------------
    # YOLO detection and crop
    # -------------------------------
    img = cv2.imread(image_path)
    results = yolo_model(img)

    if len(results[0].boxes) == 0:
        print("No banknote detected in the image.")
        exit()

    box = results[0].boxes.xyxy[0].tolist()
    x1, y1, x2, y2 = map(int, box)
    crop = img[y1:y2, x1:x2]

    # -------------------------------
    # CNN clasification
    # -------------------------------
    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    input_tensor = transform(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = cnn_model(input_tensor)
        _, pred = torch.max(output, 1)
        print(pred)
        pred_class = CLASS_NAMES[pred.item()]

    # -------------------------------
    # Show banknote with predicted class on image 
    # -------------------------------
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, pred_class, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    print("Banknote:"+ pred_class)


    img_resized = cv2.resize(img, (800, 800))
    cv2.imshow("Prediction", img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image_path = input("Enter the absolute path to the image ('exit'): ").strip()
