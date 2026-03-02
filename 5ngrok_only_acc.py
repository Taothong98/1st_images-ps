from pyngrok import ngrok
import os

# ใช้ os.environ ดึงค่าจากที่เรา export ไว้ใน Terminalฃ
# NGROK_TOKEN = "ใส่_TOKEN_จริงๆ_ของคุณที่นี่"
token = os.environ.get('NGROKKEY')

if token:
    ngrok.set_auth_token(token)
    # 2. สร้าง Tunnel
    public_url = ngrok.connect(8000)
    print(f"✅ API Public URL: {public_url}")
else:
    print("❌ ไม่พบ NGROKKEY ใน Environment Variables")


# export NGROKKEY="ใส่_TOKEN_จริงๆ_ของคุณที่นี่"
# export NGROKKEY="xxx-key"
# NGROKKEY="xxx-key" python3 5ngrok_acc.py