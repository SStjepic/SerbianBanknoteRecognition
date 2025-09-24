import os
import shutil
import random

src_dir = "./data/processed/cnn/cnn_sorted"

dst_dir = "./data/processed/cnn"
train_dir = os.path.join(dst_dir, "train")
val_dir   = os.path.join(dst_dir, "val")
test_dir  = os.path.join(dst_dir, "test")
random.seed(22)
train_ratio = 0.7
val_ratio   = 0.15
test_ratio  = 0.15

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for label in os.listdir(src_dir):
    class_folder = os.path.join(src_dir, label)
    if not os.path.isdir(class_folder):
        continue

    files = [f for f in os.listdir(class_folder) if f.endswith(".jpg")]
    random.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    train_files = files[:n_train]
    val_files   = files[n_train:n_train+n_val]
    test_files  = files[n_train+n_val:] 

    train_label_dir = os.path.join(train_dir, label)
    os.makedirs(train_label_dir, exist_ok=True)
    for f in train_files:
        shutil.copy(os.path.join(class_folder, f), os.path.join(train_label_dir, f))

    val_label_dir = os.path.join(val_dir, label)
    os.makedirs(val_label_dir, exist_ok=True)
    for f in val_files:
        shutil.copy(os.path.join(class_folder, f), os.path.join(val_label_dir, f))

    test_label_dir = os.path.join(test_dir, label)
    os.makedirs(test_label_dir, exist_ok=True)
    for f in test_files:
        shutil.copy(os.path.join(class_folder, f), os.path.join(test_label_dir, f))

print("✅ Dataset has been split into train, val, and test folders!")
