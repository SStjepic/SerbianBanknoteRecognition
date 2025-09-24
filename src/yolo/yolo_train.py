from ultralytics import YOLO

DATA_YAML = "configs/data.yaml"

model = YOLO("yolov8n.pt")  

model.train(
    data=DATA_YAML,   
    epochs=45,        
    imgsz=512,        
    batch=4,          
    name="serbian_banknotes",  
    project="runs/train", 
    exist_ok=True,
)

print("✅ Training completed!")
