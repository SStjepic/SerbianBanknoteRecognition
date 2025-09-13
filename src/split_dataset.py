import os
import shutil
import random

IMAGES_DIR = "data/processed/images"
LABELS_DIR = "data/raw/labels"

OUTPUT_DIR = "data/processed"
random.seed(22)
train_pct = 0.7
val_pct = 0.15
test_pct = 0.15

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

all_images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".jpg")]
random.shuffle(all_images)

n_total = len(all_images)
n_train = int(n_total * train_pct)
n_val = int(n_total * val_pct)

train_images = all_images[:n_train]
val_images = all_images[n_train:n_train+n_val]
test_images = all_images[n_train+n_val:]

def copy_files(image_list, split_name):
    for img in image_list:
        src_img = os.path.join(IMAGES_DIR, img)
        dst_img = os.path.join(OUTPUT_DIR, split_name, "images", img)
        shutil.copy2(src_img, dst_img)

        label_file = os.path.splitext(img)[0] + ".txt"
        src_label = os.path.join(LABELS_DIR, label_file)
        if os.path.exists(src_label):
            dst_label = os.path.join(OUTPUT_DIR, split_name, "labels", label_file)
            shutil.copy2(src_label, dst_label)

copy_files(train_images, "train")
copy_files(val_images, "val")
copy_files(test_images, "test")

print("✅ Dataset podeljen na train/val/test foldere.")
print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
