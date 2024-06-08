
from datetime import datetime, timezone
import pandas as pd
import joblib

def temp_pred(date_time):
    #load the model
    model = joblib.load('models/temperature_model.joblib')

    # Predict temperature for a new data point
    #current_time = datetime.now(timezone.utc)

    date_time =  datetime.fromisoformat(date_time).timestamp()
    print(date_time)
    #diff = (current_time-date_time).total_seconds()/3600
    predicted_temperature = model.predict([[date_time]])
    return predicted_temperature[0]



print(temp_pred("2024-09-01T00:00:00Z"))
#2023-01-20 01:16:00 