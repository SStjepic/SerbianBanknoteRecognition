from ultralytics import YOLO

model = YOLO(r"..\runs\train\serbian_banknotes\weights\best.pt")

metrics = model.val()

print("=== Evaluation Metrics ===")
print(f"Precision (P):   {metrics.box.p.mean():.3f}")
print(f"Recall (R):      {metrics.box.r.mean():.3f}")
print(f"mAP@0.5:         {metrics.box.map50:.3f}")
print(f"mAP@0.5:0.95:    {metrics.box.map:.3f}")

