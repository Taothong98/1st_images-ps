import os
import io
import base64
import json
import nest_asyncio
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pyngrok import ngrok
from ultralytics import YOLO
from PIL import Image, ImageDraw

nest_asyncio.apply()
app = FastAPI()

# สร้างโฟลเดอร์สำหรับเก็บ Template HTML
if not os.path.exists("templates"):
    os.makedirs("templates")

# --- โค้ด HTML สำหรับหน้าเว็บ ---
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sheep Detection System</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen p-8">
    <div class="max-w-4xl mx-auto bg-white rounded-xl shadow-md overflow-hidden p-6">
        <h1 class="text-3xl font-bold text-gray-800 mb-6 text-center">🐑 Sheep Detection Dashboard</h1>
        
        <form action="/predict-web" method="post" enctype="multipart/form-data" class="mb-8 border-2 border-dashed border-blue-300 p-6 rounded-lg text-center">
            <input type="file" name="file" accept="image/*" class="mb-4 block mx-auto">
            <button type="submit" class="bg-blue-600 text-white px-6 py-2 rounded-full hover:bg-blue-700 transition">Analyze Image</button>
        </form>

        {% if result_image %}
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
                <h2 class="text-xl font-semibold mb-4 text-blue-600">Detection Result</h2>
                <img src="data:image/jpeg;base64,{{ result_image }}" class="w-full rounded-lg shadow-lg border">
            </div>
            <div class="bg-gray-50 p-4 rounded-lg">
                <h2 class="text-xl font-semibold mb-4 text-green-600">JSON Metadata</h2>
                <textarea class="w-full h-64 p-2 text-xs font-mono bg-white border rounded" readonly>{{ json_data }}</textarea>
                <a href="data:text/json;charset=utf-8,{{ json_url_encoded }}" download="result.json" 
                   class="mt-4 inline-block bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 w-full text-center">
                   📥 Download JSON Result
                </a>
            </div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

with open("templates/index.html", "w") as f:
    f.write(html_content)

templates = Jinja2Templates(directory="templates")
model = YOLO('/content/runs/detect/Scenario8/weights/best.pt') 

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict-web")
async def predict_web(request: Request, file: UploadFile = File(...)):
    # 1. อ่านรูปภาพ
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 2. ประมวลผลด้วย YOLO
    results = model.predict(source=image, conf=0.25)
    
    # 3. วาด Bounding Box ลงบนรูป
    draw = ImageDraw.Draw(image)
    detections = []
    for r in results:
        for box in r.boxes:
            b = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            conf = float(box.conf)
            cls = model.names[int(box.cls)]
            draw.rectangle(b, outline="red", width=5)
            draw.text((b[0], b[1]-10), f"{cls} {conf:.2f}", fill="red")
            detections.append({"class": cls, "confidence": conf, "bbox": b})

    # 4. แปลงรูปเป็น Base64 เพื่อแสดงบนหน้าเว็บ
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    json_str = json.dumps(detections, indent=4)
    import urllib.parse
    json_url_encoded = urllib.parse.quote(json_str)

    return templates.TemplateResponse("index.html", {
        "request": request, 
        "result_image": img_str, 
        "json_data": json_str,
        "json_url_encoded": json_url_encoded
    })

# --- ส่วนรันระบบ ---
if __name__ == "__main__":
    token = os.environ.get('NGROKKEY')
    if token:
        ngrok.set_auth_token(token)
        public_url = ngrok.connect(8000)
        print(f"\n🌍 YOUR WEB APP IS LIVE AT: {public_url}")
        uvicorn.run(app, host="0.0.0.0", port=8000)