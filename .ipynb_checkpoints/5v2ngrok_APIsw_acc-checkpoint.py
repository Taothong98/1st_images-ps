import os
import nest_asyncio
import uvicorn
from fastapi import FastAPI, File, UploadFile
from pyngrok import ngrok
from ultralytics import YOLO
from PIL import Image
import io

# 1. อนุญาตให้ FastAPI รันใน Colab/Notebook
nest_asyncio.apply()

# 2. สร้าง FastAPI App
app = FastAPI()

# 3. โหลด Model (เลือกตัวที่เก่งที่สุดของคุณ)
model = YOLO('/content/runs/detect/Scenario8/weights/best.pt') 

@app.get("/")
def home():
    return {"status": "Online", "message": "Sheep Detection API is ready!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes))
    results = model.predict(source=img, conf=0.25)
    
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": [float(x) for x in box.xyxy[0]]
            })
    return {"detections": detections}

# --- ส่วนของการรัน Ngrok และ Server ---
if __name__ == "__main__":
    # ดึง Token (ที่เราส่งมาจาก Terminal)
    token = os.environ.get('NGROKKEY')
    if not token:
        print("❌ Error: Please provide NGROKKEY")
    else:
        ngrok.set_auth_token(token)
        
        # เปิดอุโมงค์
        public_url = ngrok.connect(8000)
        print(f"\n🚀 API is LIVE at: {public_url}")
        print("💡 Copy URL ด้านบนไปใส่ในเบราว์เซอร์เพื่อเช็คสถานะได้เลย")
        
        # รัน Server ค้างไว้ (โปรแกรมจะค้างอยู่ที่บรรทัดนี้ ไม่หลุด Offline)
        uvicorn.run(app, host="0.0.0.0", port=8000)