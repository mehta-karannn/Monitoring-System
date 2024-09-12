from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import base64
import time

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "FastAPI backend for heart rate and SpO2 monitoring"}

# Heart rate and SpO2 estimation function
def estimate_heart_rate_spo2(roi, fps):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    intensity = np.mean(thresh)
    heart_rate = intensity / fps * 12  # Estimated heart rate logic from intensity
    
    blue_channel, green_channel, red_channel = cv2.split(roi)
    mean_red = np.mean(red_channel)
    mean_infrared = np.mean(green_channel)
    ratio = mean_red / mean_infrared
    spo2 = -45.060 * ratio * ratio + 30.354 * ratio + 100.845  # SpO2 calculation
    
    return heart_rate, spo2

# Helper function to extract region of interest (ROI) from frame
def get_roi(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in faces:
        roi = frame[y:y + h, x:x + w]
        return roi
    
    return None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # Receive frame data as base64 from the client
            data = await websocket.receive_text()

            # Decode base64 image
            np_arr = np.frombuffer(base64.b64decode(data), np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Extract ROI and perform heart rate & SpO2 estimation
            roi = get_roi(frame)
            if roi is not None:
                fps = 30  # Set a default FPS value
                heart_rate, spo2 = estimate_heart_rate_spo2(roi, fps)

                # Send back the heart rate and SpO2 values to the client
                await websocket.send_text(f"HeartRate: {heart_rate:.0f} bpm, SpO2: {spo2:.0f}%")
            else:
                await websocket.send_text("No face detected")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
