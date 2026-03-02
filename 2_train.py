import os
import pandas as pd
from ultralytics import YOLO

# 1. โหลดโมเดล
model = YOLO('yolov8n.pt') 

# 2. เริ่มเทรน
results = model.train(
    data="/content/aerial-sheep-2/data.yaml",
    epochs=50,
    imgsz=640,
    plots=True
)

# 3. จัดการไฟล์ CSV
# results.save_dir จะเก็บ Path เช่น 'runs/detect/train4'
save_dir = results.save_dir
csv_path = os.path.join(save_dir, "results.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip() # ลบช่องว่างในชื่อหัวตาราง
    df.to_csv("summary_results.csv", index=False)
    print("สำเร็จ! เซฟไฟล์ summary_results.csv แล้ว")