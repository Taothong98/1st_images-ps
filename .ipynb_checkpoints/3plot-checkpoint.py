import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
CSV_FILE = "all_scenarios_comparison.csv"
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# อ่านข้อมูล
if not os.path.exists(CSV_FILE):
    print(f"❌ ไม่พบไฟล์ {CSV_FILE} กรุณารันโค้ดเทรนสรุปผลก่อนครับ")
    exit()

df = pd.read_csv(CSV_FILE)

# รายชื่อ Metric ที่เราต้องการพลอตแยกรูป
metrics_to_plot = {
    "mAP50": "Mean Average Precision (mAP@50)",
    "Metrics_P": "Precision",
    "Metrics_R": "Recall",
    "mAP50-95": "mAP (50-95)"
}

# ตั้งค่า Style ของกราฟ
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

for column, title in metrics_to_plot.items():
    if column not in df.columns:
        continue
        
    plt.figure(figsize=(12, 7))
    
    # สร้าง Bar Chart
    ax = sns.barplot(
        x="Scenario", 
        y=column, 
        data=df, 
        palette="viridis",
        hue="Scenario",
        legend=False
    )
    
    # ใส่ตัวเลขบนหัวแท่งกราฟ (เพื่อให้ดูค่าได้แม่นยำใน Paper)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=10,
                    fontweight='bold')

    plt.title(f"Comparison of {title} across Scenarios", fontsize=16, pad=20)
    plt.xlabel("Experimental Scenarios", fontsize=12)
    plt.ylabel("Score (0.0 - 1.0)", fontsize=12)
    plt.ylim(0, 1.1) # ปรับสเกลให้เห็นความแตกต่างชัดเจน
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # เซฟรูปลงในโฟลเดอร์ plots
    file_path = os.path.join(OUTPUT_DIR, f"comparison_{column}.png")
    plt.savefig(file_path, dpi=300) # dpi=300 สำหรับความคมชัดระดับสิ่งพิมพ์
    plt.close()
    
    print(f"✅ สร้างกราฟสำเร็จ: {file_path}")

print(f"\n🚀 พลอตกราฟครบทุกเมทริกซ์แล้ว! ตรวจสอบได้ที่โฟลเดอร์: {OUTPUT_DIR}")