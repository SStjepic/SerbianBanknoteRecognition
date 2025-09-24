import os
from PIL import Image

RAW_DIR = "data/raw/images"

PROCESSED_DIR = "data/processed/images"
os.makedirs(PROCESSED_DIR, exist_ok=True)

SIZE = (512, 512)
counter = 1
dataset = os.listdir(RAW_DIR)
dataset_len = len(dataset)
for filename in dataset:
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(RAW_DIR, filename)
        img = Image.open(img_path)

        img_resized = img.resize(SIZE, Image.LANCZOS)

        save_path = os.path.join(PROCESSED_DIR, filename)
        img_resized.save(save_path)

        print(f'{counter}/{dataset_len}')
        counter += 1

print("✅ All images have been scaled to 512x512 and saved in", PROCESSED_DIR)
