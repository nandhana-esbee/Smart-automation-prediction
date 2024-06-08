from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
df = pd.read_csv('data/Temperature.csv')
#print(df)

from sklearn.linear_model import LinearRegression

# Sample data (replace with your actual data)
df['_time'] = pd.to_datetime(df["_time"])
df['Time_Interval'] = df['_time'].diff().dt.total_seconds() / 3600  
#print(df["Time_Interval"])

X = df["Time_Interval"]  # Features (e.g., time of day))
y = df["_value"]      # Target values (e.g., temperature)
datafit = pd.DataFrame(dict(x=X, y=y))
datafit = datafit.dropna()
#print(df.dtypes)

# #Create and fit the linear regression model
model = LinearRegression()
model.fit(datafit[["x"]], datafit["y"])
print("success")
# # Predict temperature for a new data point
# df["_start"] = pd.to_datetime(df["_start"])
# for time in df["_start"]:
#    new_data = time
#    #print(new_data)
#    predicted_temperature = model.predict(new_data)

#    # Print the predicted temperature
#    print(f"Predicted temperature for time {new_data} :{predicted_temperature}")



# # Save the model
# from sklearn.pipeline import Pipeline
# import pickle
# Tfidf = TfidfVectorizer()
# GB = GradientBoostingClassifier()
# pipe = Pipeline([("Tfidf", Tfidf), ("GB", GB)])
# pipe.fit(X, y)

# with open('pipe.pickle', 'wb') as picklefile:
#     pickle.dump(pipe, picklefile)
