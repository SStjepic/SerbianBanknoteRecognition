import os
import shutil

src_dir = "./data/processed/cnn/cnn_dataset"       
dst_dir = "./data/processed/cnn/cnn_sorted"       

os.makedirs(dst_dir, exist_ok=True)

denominations = ["10", "20", "50", "100", "200", "500", "1000", "2000", "5000"]
denom_map = {d: f"{i}_{d}" for i, d in enumerate(denominations)}

for file in os.listdir(src_dir):
    if file.endswith(".jpg"):
        denom = file.split("RSD")[0].strip()

        label_dir = os.path.join(dst_dir, denom_map[denom])
        os.makedirs(label_dir, exist_ok=True)

        shutil.copy(
            os.path.join(src_dir, file),
            os.path.join(label_dir, file)
        )

print("✅ All images have been sorted into numbered denomination folders!")
