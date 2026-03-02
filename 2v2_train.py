import os
import yaml
import pandas as pd
from ultralytics import YOLO

# --- CONFIGURATION ---
BASE_PATH = "/content/aerial-sheep-2"
YAML_TEMPLATE = os.path.join(BASE_PATH, "data.yaml")

# นิยาม Scenario ตามที่คุณกำหนด
scenarios = {
    "Scenario1": ["train"],
    "Scenario2": ["train_augmented"],
    "Scenario3": ["train_color_light_aug"],
    "Scenario4": ["train_enhanced"],
    "Scenario5": ["train", "train_augmented"],
    "Scenario6": ["train", "train_color_light_aug"],
    "Scenario7": ["train", "train_enhanced"],
    "Scenario8": ["train", "train_augmented", "train_color_light_aug", "train_enhanced"]
}

# อ่านไฟล์ data.yaml ต้นฉบับ
with open(YAML_TEMPLATE, 'r') as f:
    data_config = yaml.safe_load(f)

# ตัวแปรสำหรับเก็บข้อมูลสรุปภาพรวม
comparison_data = []

# --- LOOP TRAINING ---
for name, folders in scenarios.items():
    print(f"\n--- 🚀 เริ่มการทดลอง: {name} ---")
    
    # 1. เตรียม Path ของรูปภาพ (รวมหลายโฟลเดอร์สำหรับ Scenario 5-8)
    train_paths = [os.path.join(BASE_PATH, f, "images") for f in folders]
    data_config['train'] = train_paths
    data_config['val'] = os.path.join(BASE_PATH, "test/images") # ใช้ Test เป็น Val ตามที่คุณตั้งค่าไว้
    
    # 2. สร้างไฟล์ YAML ชั่วคราวสำหรับ Scenario นี้
    temp_yaml = f"data_{name}.yaml"
    with open(temp_yaml, 'w') as f:
        yaml.dump(data_config, f)

    # 3. เริ่มเทรน (Epochs=50 ตามที่คุณต้องการ)
    model = YOLO('yolov8n.pt') 
    results = model.train(
        data=temp_yaml,
        epochs=50,
        imgsz=640,
        name=name,    # ผลลัพธ์จะอยู่ใน runs/detect/ScenarioX
        device=0,
        verbose=False
        # save=False,     # <--- เพิ่มตรงนี้ ถ้าไม่อยากให้เซฟไฟล์ .pt (weights)
        # plots=False     # <--- เพิ่มตรงนี้ ถ้าไม่อยากให้เซฟรูปกราฟต่างๆ ลงดิสก์
    )

    # 4. จัดการไฟล์ CSV ของ Scenario นี้ (Summary ราย Epoch)
    src_csv = os.path.join(results.save_dir, "results.csv")
    if os.path.exists(src_csv):
        df_epoch = pd.read_csv(src_csv)
        df_epoch.columns = df_epoch.columns.str.strip()
        # เซฟแยกเป็นไฟล์เฉพาะของ Scenario นั้นๆ
        df_epoch.to_csv(f"summary_{name}.csv", index=False)
        print(f"✅ เซฟไฟล์ราย Epoch: summary_{name}.csv")

        # 5. ดึงค่าจากแถวสุดท้าย (Best Epoch) มาเก็บในตารางเปรียบเทียบ
        last_metrics = df_epoch.iloc[-1]
        comparison_data.append({
            "Scenario": name,
            "Folders": "+".join(folders),
            "Train_Box_Loss": last_metrics.get("train/box_loss", 0),
            "Train_Cls_Loss": last_metrics.get("train/cls_loss", 0),
            "Metrics_P": last_metrics.get("metrics/precision(B)", 0),
            "Metrics_R": last_metrics.get("metrics/recall(B)", 0),
            "mAP50": last_metrics.get("metrics/mAP50(B)", 0),
            "mAP50-95": last_metrics.get("metrics/mAP50-95(B)", 0)
        })

# --- 6. สร้างไฟล์สรุปเปรียบเทียบทุก Scenario ---
comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv("all_scenarios_comparison.csv", index=False)

print("\n" + "="*50)
print("📊 การทดลองเสร็จสิ้นสมบูรณ์!")
print("ไฟล์ที่สร้างขึ้น:")
print("1. summary_Scenario1.csv ถึง Scenario8.csv (รายละเอียดราย Epoch)")
print("2. all_scenarios_comparison.csv (ตารางเปรียบเทียบประสิทธิภาพ)")
print("="*50)
print(comparison_df[['Scenario', 'mAP50', 'Metrics_P', 'Metrics_R']])