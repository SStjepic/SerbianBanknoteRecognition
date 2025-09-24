from ultralytics import YOLO
import glob
import cv2
import os

model = YOLO(r"..\runs\train\serbian_banknotes\weights\best.pt")  

image_folder = "data/processed/val/images"
image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))

for img_path in image_paths:
    img = cv2.imread(img_path)
    
    results = model(img)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0] 
            cls = box.cls[0]

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"Banknote {conf:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    scale = 800 / img.shape[1]
    new_dim = (800, int(img.shape[0] * scale))
    resized_img = cv2.resize(img, new_dim)

    cv2.imshow("Banknote", resized_img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
