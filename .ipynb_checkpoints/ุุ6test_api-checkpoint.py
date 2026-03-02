import requests

# 1. เปลี่ยนเป็น URL ที่ Ngrok ให้มา
url = "https://mistakably-transmental-vilma.ngrok-free.dev/predict"

# 2. ระบุไฟล์ภาพที่คุณต้องการทดสอบ
image_path = "test_sheep.jpg" 

with open(image_path, 'rb') as f:
    files = {'file': f}
    # ยิง Request แบบ POST
    response = requests.post(url, files=files)

# 3. ดูผลลัพธ์
if response.status_code == 200:
    results = response.json()
    print("🎯 ตรวจพบวัตถุ:")
    for det in results['detections']:
        print(f"- {det['class']} (มั่นใจ: {det['confidence']:.2f})")
        print(f"  ตำแหน่ง Bbox: {det['bbox']}")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)