from ultralytics import YOLO
import cv2
import os

model = YOLO(r"..\runs\train\serbian_banknotes\weights\best.pt")
src_dir = "data/raw/images"
dst_dir = "data/processed/cnn/cnn_dataset"

os.makedirs(dst_dir, exist_ok=True)
counter = 1
dataset = os.listdir(src_dir)
dataset_len = len(dataset)

img_size = 256

for img_file in dataset:
    path = os.path.join(src_dir, img_file)
    results = model(path)
    img = cv2.imread(path)
    
    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = img[y1:y2, x1:x2]
        
        crop_resized = cv2.resize(crop, (img_size, img_size))
        
        out_path = os.path.join(dst_dir, f"{os.path.splitext(img_file)[0]}_{i}.jpg")
        cv2.imwrite(out_path, crop_resized)
        
        print(f'{counter}/{dataset_len}')
        counter += 1
