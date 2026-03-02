import albumentations as A
import cv2
import os
import glob
from tqdm import tqdm

# ==================== CONFIGURATION ====================
DATASET_ROOT = "aerial-sheep-2"
SPLIT = "train"                               # เปลี่ยนเป็น "valid" หรือ "test" ได้

INPUT_IMAGES_DIR = os.path.join(DATASET_ROOT, SPLIT, "images")
INPUT_LABELS_DIR  = os.path.join(DATASET_ROOT, SPLIT, "labels")

OUTPUT_DIR = os.path.join(DATASET_ROOT, f"{SPLIT}_color_light_aug")
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_LABELS_DIR  = os.path.join(OUTPUT_DIR, "labels")

AUGMENT_PER_IMAGE = 1                         # จำนวนรูปใหม่ต่อ 1 รูปต้นฉบับ

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

# ==================== COLOR & LIGHT TRANSFORMATIONS ====================
# เน้นปรับแสง สี ความคมชัด และ grayscale โดยไม่มีการหมุน/พลิก
color_light_transform = A.Compose([
    # Grayscale (ทำให้ภาพขาวดำ) – ช่วยให้โมเดลโฟกัสที่ shape มากกว่าสี
    A.ToGray(p=0.25),  # ปรับ p ได้ตามต้องการ (0.2–0.4 แนะนำ)

    # Brightness & Contrast – จำลองแสงแดดจ้า / มืดครึ้ม / หมอก
    A.RandomBrightnessContrast(
        brightness_limit=0.35,     # ±35% ความสว่าง
        contrast_limit=0.35,       # ±35% ความคมชัด
        p=0.7
    ),

    # Color Jittering – กวน Hue, Saturation, Value
    A.HueSaturationValue(
        hue_shift_limit=25,        # ±25 หน่วย hue (เปลี่ยนโทนสี)
        sat_shift_limit=40,        # ±40% saturation (ความสดของสี)
        val_shift_limit=30,        # ±30% value (ความสว่างโดยรวม)
        p=0.6
    ),

    # RGB Shift – กวนสีแต่ละ channel แยก (จำลอง white balance เปลี่ยน)
    A.RGBShift(
        r_shift_limit=25,
        g_shift_limit=25,
        b_shift_limit=25,
        p=0.5
    ),

    # เพิ่ม noise เล็กน้อย (จำลองภาพจากกล้องจริงที่มี noise)
    A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),

    # อาจเพิ่ม RandomGamma ถ้าต้องการจำลอง exposure ที่ต่างกัน
    A.RandomGamma(gamma_limit=(70, 130), p=0.4),

], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_visibility=0.2,          # ถ้า bbox เหลือ <20% ให้ตัดทิ้ง
    min_area=1e-5,
    clip=True                    # คลิป bbox ให้อยู่ในภาพ (ป้องกัน error)
))

# ==================== HELPER ====================
def get_base_name(filename):
    return os.path.splitext(filename)[0]

def is_valid_bbox(bbox):
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return False
    x_min = x - w / 2
    x_max = x + w / 2
    y_min = y - h / 2
    y_max = y + h / 2
    if x_max <= x_min or y_max <= y_min:
        return False
    if not (0 <= x_min <= 1 and 0 <= x_max <= 1 and 0 <= y_min <= 1 and 0 <= y_max <= 1):
        return False
    return True

# ==================== MAIN ====================
image_files = sorted(glob.glob(os.path.join(INPUT_IMAGES_DIR, "*.jpg")))

print(f"พบภาพทั้งหมด: {len(image_files):,} ภาพ")
print(f"จะสร้าง ≈ {len(image_files) * AUGMENT_PER_IMAGE:,} รูป (color & light only)\n")

skipped = 0
processed = 0
created = 0

for img_path in tqdm(image_files, desc="Color & Light Augment"):
    filename = os.path.basename(img_path)
    base_name = get_base_name(filename)
    label_path = os.path.join(INPUT_LABELS_DIR, base_name + ".txt")

    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ ข้าม: {filename} → อ่านภาพไม่ได้")
        skipped += 1
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # อ่าน label
    bboxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        cls = int(parts[0])
                        bbox = [float(x) for x in parts[1:]]
                        if is_valid_bbox(bbox):
                            class_labels.append(cls)
                            bboxes.append(bbox)
                    except:
                        pass

    if not bboxes:
        print(f"⚠️ ข้าม: {filename} → ไม่มี bbox ที่ถูกต้อง")
        skipped += 1
        continue

    # สร้าง augmented versions
    for i in range(AUGMENT_PER_IMAGE):
        transformed = color_light_transform(image=image, bboxes=bboxes, class_labels=class_labels)

        aug_image = transformed['image']
        aug_bboxes = transformed['bboxes']
        aug_classes = transformed['class_labels']

        if not aug_bboxes:
            continue

        new_name = f"{base_name}_color_aug_{i}"
        save_img = os.path.join(OUTPUT_IMAGES_DIR, f"{new_name}.jpg")
        save_txt = os.path.join(OUTPUT_LABELS_DIR, f"{new_name}.txt")

        cv2.imwrite(save_img, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

        with open(save_txt, 'w') as f:
            for cls, bbox in zip(aug_classes, aug_bboxes):
                f.write(f"{cls} {' '.join(f'{x:.6f}' for x in bbox)}\n")

        created += 1

    processed += 1

# ==================== SUMMARY ====================
print("\n" + "="*70)
print("เสร็จสิ้น Color & Light Augmentation")
print(f"ประมวลผลสำเร็จ     : {processed:,} ภาพ")
print(f"สร้างรูปใหม่สำเร็จ  : {created:,} รูป")
print(f"ข้ามภาพ             : {skipped:,} ภาพ")
print(f"ผลลัพธ์ → {OUTPUT_DIR}")
print("="*70)