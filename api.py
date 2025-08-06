from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class AlertRequest(BaseModel):
    names: List[str]

@app.post("/alert")
def send_alert(request: AlertRequest):
    print(f"⚠️ ALERT: Identified persons - {', '.join(request.names)}")
    return {"message": "Criminal/Missing Person Identified!", "persons": request.names}

@app.get("/target_face_alert")
def send_target_face_alert():
    print(f"⚠️ ALERT: Missing Target detected!!")
    return {"message": "🚨 Missing Target detected!"}

# API endpoint to send an alert when a weapon is detected
@app.get("/weapon_alert")
def send_weapon_alert():
    print(f"⚠️ ALERT: Detected Weapons!!")
    return {"alert": "Weapon detected! Security alert triggered!"}

@app.get("/track_alert")
def track_alert():
    print(f"⚠️ ALERT: Detected Passing through Railway Tracks!!")
    return {"alert": "Person on railway track! Emergency alert triggered!"}

@app.get("/crime_alert")
def crime_alert():
    print(f"⚠️ ALERT: Criminal Activity Detected!!")
    return {"alert": "Crime activity detected and alert triggered!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
