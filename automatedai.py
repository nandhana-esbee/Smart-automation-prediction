import pandas as pd
from datetime import datetime ,timezone

# Load data
df = pd.read_csv('data/Temperature.csv')

from sklearn.linear_model import LinearRegression

# Sample data (replace with your actual data)
df['_time'] = pd.to_datetime(df["_time"])
df['Time_Interval'] = df['_time'].diff().dt.total_seconds() / 3600  

X = df["Time_Interval"]  # Features (e.g., time of day))
y = df["_value"]      # Target values (e.g., temperature)
datafit = pd.DataFrame(dict(x=X, y=y))
datafit = datafit.dropna()


# #Create and fit the linear regression model
model = LinearRegression()
model.fit(datafit[["x"]], datafit["y"])
print("success")


# Predict temperature for a new data point
current_time = datetime.now(timezone.utc)

df["_start"] = pd.to_datetime(df["_start"])
df['time_difference_seconds'] = (current_time - df['_start']).dt.total_seconds()/3600
#print(df['time_difference_seconds'])

for time in df["time_difference_seconds"]:
    break
print(time)
print(model.predict([[time]])[0])
    #predicted_temperature = model.predict(time)
    # Print the predicted temperature
    #print(f"Predicted temperature  :{predicted_temperature}")



# # Save the model
# from sklearn.pipeline import Pipeline
# import pickle
# Tfidf = TfidfVectorizer()
# GB = GradientBoostingClassifier()
# pipe = Pipeline([("Tfidf", Tfidf), ("GB", GB)])
# pipe.fit(X, y)

# with open('pipe.pickle', 'wb') as picklefile:
#     pickle.dump(pipe, picklefile)
