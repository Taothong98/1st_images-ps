import albumentations as A
import cv2
import os
import glob
import numpy as np
from tqdm import tqdm

# ==================== CONFIG ====================
DATASET_ROOT = "aerial-sheep-2"
SPLIT = "train"                               # เปลี่ยนเป็น "valid" หรือ "test" ได้

INPUT_IMAGES_DIR = os.path.join(DATASET_ROOT, SPLIT, "images")
INPUT_LABELS_DIR  = os.path.join(DATASET_ROOT, SPLIT, "labels")

OUTPUT_DIR = os.path.join(DATASET_ROOT, f"{SPLIT}_enhanced")
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_LABELS_DIR  = os.path.join(OUTPUT_DIR, "labels")

AUGMENT_PER_IMAGE = 1                         # จำนวนเวอร์ชันใหม่ต่อภาพต้นฉบับ

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

# ==================== TRANSFORMATIONS ====================
# Denoising + Enhancement เบา ๆ (ไม่มี transform ที่กระทบตำแหน่ง bbox หนัก)
enhance_transform = A.Compose([
    # Denoising
    A.GaussianBlur(blur_limit=(3, 5), p=0.4),
    # A.MedianBlur(blur_limit=5, p=0.3),                    # ทางเลือกอื่น

    # Sharpening หลังลด noise
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.6),

    # เพิ่ม contrast ในพื้นที่ท้องถิ่น (ดีสำหรับภาพโดรน)
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),

    # Brightness/Contrast เบา ๆ
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.20, p=0.6),

    # Unsharp mask เพิ่มความคมชัด
    A.UnsharpMask(blur_limit=(3, 7), sigma_limit=0.5, alpha=(0.1, 0.3), p=0.4),

], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_visibility=0.3,
    min_area=1e-6,
    clip=True,                    # คลิป bbox ให้อยู่ในภาพเสมอ
    check_each_transform=False    # ไม่ต้องเช็คทุก transform เพราะไม่มี transform ที่เปลี่ยนตำแหน่ง
))

# ==================== HELPER FUNCTIONS ====================
def get_base_name(filename):
    return os.path.splitext(filename)[0]

def is_valid_bbox(bbox):
    """ตรวจสอบว่า bbox ใช้ได้จริงใน YOLO format"""
    try:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return False
        x_min = x - w / 2
        x_max = x + w / 2
        y_min = y - h / 2
        y_max = y + h / 2
        if x_max <= x_min + 1e-6 or y_max <= y_min + 1e-6:
            return False
        # ตรวจว่าอยู่ในช่วง 0-1
        if not (0 <= x_min <= 1 and 0 <= x_max <= 1 and 
                0 <= y_min <= 1 and 0 <= y_max <= 1):
            return False
        return True
    except:
        return False

# ==================== MAIN LOOP ====================
image_files = sorted(glob.glob(os.path.join(INPUT_IMAGES_DIR, "*.jpg")))

print(f"พบภาพทั้งหมด: {len(image_files):,} ภาพ")
print(f"จะสร้าง ≈ {len(image_files) * AUGMENT_PER_IMAGE:,} รูป enhanced\n")

skipped = 0
processed = 0
created = 0

for img_path in tqdm(image_files, desc="Enhancing images"):
    filename = os.path.basename(img_path)
    base_name = get_base_name(filename)
    label_path = os.path.join(INPUT_LABELS_DIR, base_name + ".txt")

    # อ่านภาพ
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ ข้ามภาพ: {filename} → อ่านภาพไม่ได้")
        skipped += 1
        continue
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # อ่านและกรอง label อย่างเข้มงวด
    bboxes = []
    class_labels = []
    invalid_bbox_count = 0

    try:
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    print(f"Warning: {filename} line {line_num}: {len(parts)} fields (ควรมี 5) → ข้ามบรรทัด")
                    invalid_bbox_count += 1
                    continue
                try:
                    cls = int(parts[0])
                    bbox = [float(x) for x in parts[1:]]
                    if is_valid_bbox(bbox):
                        class_labels.append(cls)
                        bboxes.append(bbox)
                    else:
                        print(f"Warning: {filename} line {line_num}: invalid bbox {bbox} → ข้าม bbox นี้")
                        invalid_bbox_count += 1
                except ValueError as ve:
                    print(f"Warning: {filename} line {line_num}: ไม่สามารถแปลงตัวเลขได้ ({ve}) → ข้ามบรรทัด")
                    invalid_bbox_count += 1
    except Exception as e:
        print(f"Error: {filename} → อ่าน label ล้มเหลว: {e}")
        skipped += 1
        continue

    # ถ้าไม่มี bbox ที่ใช้ได้เลย หรือมีปัญหาเยอะ → ข้ามทั้งภาพ
    if len(bboxes) == 0 or invalid_bbox_count > len(bboxes) * 0.5:  # ถ้าครึ่งหนึ่งผิดก็ข้าม
        reason = "ไม่มี bbox ที่ถูกต้องเหลือเลย" if len(bboxes) == 0 else f"มี bbox ผิด {invalid_bbox_count} ตัวจาก {len(bboxes)+invalid_bbox_count}"
        print(f"⚠️ ข้ามภาพทั้งหมด: {filename} → {reason}")
        skipped += 1
        continue

    # ทำ augmentation
    for i in range(AUGMENT_PER_IMAGE):
        try:
            transformed = enhance_transform(image=image_rgb, bboxes=bboxes, class_labels=class_labels)
            
            aug_image = transformed['image']
            aug_bboxes = transformed['bboxes']
            aug_classes = transformed['class_labels']

            if len(aug_bboxes) == 0:
                continue  # ข้ามเวอร์ชันนี้ถ้า bbox หายหมด

            new_name = f"{base_name}_enhanced_{i}"
            save_img_path = os.path.join(OUTPUT_IMAGES_DIR, f"{new_name}.jpg")
            save_txt_path = os.path.join(OUTPUT_LABELS_DIR, f"{new_name}.txt")

            cv2.imwrite(save_img_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

            with open(save_txt_path, 'w') as f:
                for cls, bbox in zip(aug_classes, aug_bboxes):
                    f.write(f"{cls} {' '.join(f'{coord:.6f}' for coord in bbox)}\n")

            created += 1
        except Exception as e:
            print(f"Error augmenting {filename} version {i}: {e}")
            continue

    processed += 1

# ==================== SUMMARY ====================
print("\n" + "="*80)
print("เสร็จสิ้นการ enhance / denoising")
print(f"ประมวลผลสำเร็จ (มี bbox ดีพอใช้) : {processed:,} ภาพ")
print(f"สร้างรูป enhanced สำเร็จ           : {created:,} รูป")
print(f"ข้ามทั้งภาพ                         : {skipped:,} ภาพ")
print(f"ผลลัพธ์อยู่ใน: {OUTPUT_DIR}")
print("="*80)

if skipped > 0:
    print("คำแนะนำ: ตรวจไฟล์ .txt ในโฟลเดอร์ labels ที่ถูกข้าม อาจมีบรรทัดผิดรูปแบบหรือค่า w/h = 0")