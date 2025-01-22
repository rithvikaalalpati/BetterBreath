from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Replace with production URL if applicable
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and scaler
try:
    model = pickle.load(open('air_quality.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError as e:
    raise Exception("Model or Scaler file not found. Ensure 'air_quality.pkl' and 'scaler.pkl' are in the working directory.")

# Route to handle predictions
@app.post("/predict", response_class=JSONResponse)
async def predict(
    Temperature: float = Form(...),
    Humidity: float = Form(...),
    PM25: float = Form(...),
    PM10: float = Form(...),
    SO2: float = Form(...),
    NO2: float = Form(...),
    CO: float = Form(...),
    Proximity: float = Form(...),
    Population: float = Form(...),
):
    
    logging.info(f"Received data: {Temperature}, {Humidity}, {PM25}, {PM10}, {SO2}, {NO2}, {CO}, {Proximity}, {Population}")

    """
    Predict air quality based on user inputs.
    """
    # Validate PM2.5 and PM10
    if PM25 < PM10:
        raise HTTPException(status_code=400, detail="PM2.5 must be greater than or equal to PM10.")

    # Calculate PM
    PM = PM25 - PM10

    # Prepare features
    features = pd.DataFrame(
        [[Temperature, Humidity, PM, NO2, SO2, CO, Proximity, Population]],
        columns=[
            'Temperature',
            'Humidity',
            'PM',
            'NO2',
            'SO2',
            'CO',
            'Proximity_to_Industrial_Areas',
            'Population_Density',
        ],
    )

    # Scale features
    try:
        features_scaled = scaler.transform(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error scaling input features.")

    # Predict using the model
    try:
        prediction = model.predict(features_scaled)
        prediction = int(prediction[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating prediction.")

    # Generate response message
    if prediction == 0:
        result_message = (
            "Prediction: Uh oh 😞, the Air Quality is POOR in your area. Please stay safe 🙏"
        )
    else:
        result_message = "Prediction: The Air Quality in your area looks GOOD 😊."

    return {"Prediction": result_message}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("fast:app", host="127.0.0.1", port=8000, reload=True)
