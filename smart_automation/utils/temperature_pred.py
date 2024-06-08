
from datetime import datetime, timezone , timedelta
import pandas as pd
import joblib

def temp_pred():
    '''
    This function predicts the temperature for the next hour using the trained model.
    '''
    #load the model
    model = joblib.load('trained_models/temperature_model.joblib')

    # Predict temperature for a new data point
    current_time = datetime.now(timezone.utc)
    pred_time = current_time + timedelta(hours=1)
    #date_time =  datetime.fromisoformat(pred_time)
    #print(date_time)
    diff = (current_time-pred_time).total_seconds()/3600
    predicted_temperature = model.predict([[diff]])
    return predicted_temperature[0]


print(temp_pred())
#2023-01-20 01:16:00 