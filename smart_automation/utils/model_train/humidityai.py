import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timezone

# Load data
df = pd.read_csv('data/Humidity.csv')
df = df.dropna(axis=1)

#only take _time,_value and _start columns
df = df[['_time','_value','_start']]
df = df.dropna()

df['_time'] = pd.to_datetime(df["_time"])
df['_time'] = df['_time'].dt.tz_localize(None)

df['Time_Interval'] = df['_time'].diff().dt.total_seconds() / 3600  
df['Time_Interval'] = df['Time_Interval'].fillna(0)  # Replace NaN with 0

X = df["Time_Interval"] # Features (e.g., time of day))
y = df["_value"]      # Target values (e.g., temperature)
datafit = pd.DataFrame(dict(x=X, y=y))
datafit = datafit.dropna()


# #Create and fit the linear regression model
model = LinearRegression()
model.fit(datafit[["x"]], datafit["y"])
df = df.dropna()

#save model
import joblib
joblib.dump(model, 'trained_models/humidity_model.joblib')
print("success")
