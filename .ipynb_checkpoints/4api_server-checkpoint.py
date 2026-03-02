import nest_asyncio
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import uvicorn
from PIL import Image
import io

# 1. อนุญาตให้รัน asyncio ใน Colab
nest_asyncio.apply()

app = FastAPI()

# 2. โหลด Model ที่ดีที่สุดของคุณ (เปลี่ยน Path ให้ตรงกับที่เทรนเสร็จ)
model = YOLO('/content/runs/detect/Scenario5/weights/best.pt') 

@app.get("/")
def home():
    return {"message": "Sheep Detection API is Running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # อ่านไฟล์รูปที่ส่งมา
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes))
    
    # สั่ง Model ประมวลผล
    results = model.predict(source=img, conf=0.25)
    
    # ดึงข้อมูลการตรวจจับ (Bounding Boxes)
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": [float(x) for x in box.xyxy[0]] # [x1, y1, x2, y2]
            })
            
    return JSONResponse(content={"detections": detections})