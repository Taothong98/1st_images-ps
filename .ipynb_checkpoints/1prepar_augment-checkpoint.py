import albumentations as A
import cv2
import os
import glob
from tqdm import tqdm

# ==================== CONFIGURATION ====================
DATASET_ROOT = "aerial-sheep-2"
SPLIT = "train"                               # เปลี่ยนเป็น "valid" หรือ "test" ได้
# SPLIT = "test"

INPUT_IMAGES_DIR = os.path.join(DATASET_ROOT, SPLIT, "images")
INPUT_LABELS_DIR  = os.path.join(DATASET_ROOT, SPLIT, "labels")

OUTPUT_DIR = os.path.join(DATASET_ROOT, f"{SPLIT}_augmented")
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_LABELS_DIR  = os.path.join(OUTPUT_DIR, "labels")

AUGMENT_PER_IMAGE = 1                         # จำนวนรูปใหม่ต่อ 1 รูปต้นฉบับ

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

# ==================== AUGMENTATIONS ====================
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.4),
    A.Rotate(limit=45, p=0.6),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_visibility=0.1,      # ถ้าเหลือน้อยกว่า 10% ให้ albumentations ตัด bbox นั้นทิ้ง
    min_area=1e-5,
    check_each_transform=True
))

# ==================== HELPER ====================
def get_base_name(filename):
    return os.path.splitext(filename)[0]

def is_valid_bbox(bbox):
    """ตรวจว่า bbox ใช้ได้จริงหรือไม่ (สำหรับ YOLO format)"""
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return False
    x_min = max(0, x - w / 2)
    x_max = min(1, x + w / 2)
    y_min = max(0, y - h / 2)
    y_max = min(1, y + h / 2)
    if x_max <= x_min or y_max <= y_min:
        return False
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return False
    return True

# ==================== MAIN ====================
image_files = sorted(glob.glob(os.path.join(INPUT_IMAGES_DIR, "*.jpg")))

print(f"พบภาพทั้งหมด: {len(image_files):,} ภาพ")
print(f"จะสร้าง ≈ {len(image_files) * AUGMENT_PER_IMAGE:,} รูป augmented\n")

skipped_images = 0
processed_images = 0
total_aug_created = 0

for img_path in tqdm(image_files, desc="Augmenting"):
    filename = os.path.basename(img_path)
    base_name = get_base_name(filename)
    label_path = os.path.join(INPUT_LABELS_DIR, base_name + ".txt")

    # อ่านภาพ
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ ข้ามภาพ: {filename} → อ่านภาพไม่ได้")
        skipped_images += 1
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # อ่าน label และตรวจสอบ
    bboxes = []
    class_labels = []
    invalid_count = 0

    try:
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) != 5:
                    invalid_count += 1
                    continue
                cls = int(parts[0])
                bbox = [float(x) for x in parts[1:]]
                if is_valid_bbox(bbox):
                    class_labels.append(cls)
                    bboxes.append(bbox)
                else:
                    invalid_count += 1
    except Exception as e:
        print(f"⚠️ ข้ามภาพ: {filename} → อ่าน label ล้มเหลว ({str(e)})")
        skipped_images += 1
        continue

    # ถ้ามี bbox ผิด หรือไม่มี bbox ที่ใช้ได้เลย → ข้ามทั้งภาพ
    if invalid_count > 0 or not bboxes:
        reason = "ไม่มี bbox ที่ถูกต้องเหลือเลย" if not bboxes else f"มี bbox ผิด {invalid_count} ตัว"
        print(f"⚠️ ข้ามภาพทั้งหมด: {filename} → {reason}")
        skipped_images += 1
        continue

    # ทำ Augmentation
    for i in range(AUGMENT_PER_IMAGE):
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        
        aug_image = transformed['image']
        aug_bboxes = transformed['bboxes']
        aug_classes = transformed['class_labels']

        # หลัง augment ถ้า bbox หายหมด → ข้ามเวอร์ชันนี้ (แต่ยังทำเวอร์ชันอื่นต่อ)
        if not aug_bboxes:
            continue

        new_name = f"{base_name}_aug_{i}"
        save_img = os.path.join(OUTPUT_IMAGES_DIR, f"{new_name}.jpg")
        save_txt = os.path.join(OUTPUT_LABELS_DIR, f"{new_name}.txt")

        cv2.imwrite(save_img, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

        with open(save_txt, 'w') as f:
            for cls, bbox in zip(aug_classes, aug_bboxes):
                f.write(f"{cls} {' '.join(f'{x:.6f}' for x in bbox)}\n")

        total_aug_created += 1

    processed_images += 1

# ==================== SUMMARY ====================
print("\n" + "="*80)
print("✅ เสร็จสิ้น!")
print(f"ประมวลผลสำเร็จ (มี bbox ดีทั้งหมด) : {processed_images:,} ภาพ")
print(f"สร้างรูป augmented สำเร็จ             : {total_aug_created:,} รูป")
print(f"ข้ามทั้งภาพ (bbox ผิด/ไม่มี/อ่านไม่ได้) : {skipped_images:,} ภาพ")
print(f"ผลลัพธ์อยู่ใน: {OUTPUT_DIR}")
print("="*80)

if skipped_images > 0:
    print("แนะนำ: ตรวจ label ในโฟลเดอร์ labels อีกครั้ง เพราะมีภาพหลายรูปที่มีปัญหา bbox")
    
# # คัดลอกรูปภาพเดิมไปรวม
# cp Aerial-Sheep-1/train/images/*.jpg Aerial-Sheep-1/train_augmented/images/
# # คัดลอก Label เดิมไปรวม
# cp Aerial-Sheep-1/train/labels/*.txt Aerial-Sheep-1/train_augmented/labels/
# pip3 install -U albumentations opencv-python-headless
