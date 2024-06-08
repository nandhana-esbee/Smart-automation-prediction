from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
df = pd.read_csv('data/Temperature.csv')
print(df)

from sklearn.linear_model import LinearRegression

# Sample data (replace with your actual data)
X =pd.to_datetime(df["_time"]) # Features (e.g., time of day)
y = df["_value"]      # Target values (e.g., temperature)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict temperature for a new data point (time = 6)
df["_start"] = pd.to_datetime(df["_start"])
for time in df["_start"]:
    new_data = time
predicted_temperature = model.predict(new_data)

# Print the predicted temperature
print(f"Predicted temperature for time {new_data} :{predicted_temperature}")

# X = df['Text_preprocess']
# y = df['label']

# # Split data into training and testing sets (optional, adjust test_size as needed)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Feature extraction (using TF-IDF in this example)
# vectorizer = TfidfVectorizer(max_features=1000)  # Hyperparameter: number of features
# X_train_features = vectorizer.fit_transform(X_train)
# X_test_features = vectorizer.transform(X_test)

# # Model training 
# model = GradientBoostingClassifier(n_estimators=5000, learning_rate=0.05, max_depth=3 , max_features='log2',verbose=1)

# time1 = time.time()

# # Train the model
# model.fit(X_train_features, y_train )

# # Model evaluation 
# from sklearn.metrics import accuracy_score ,f1_score, precision_score, recall_score
# y_pred = model.predict(X_test_features)
# accuracy = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred, average='weighted')
# precision = precision_score(y_test, y_pred, average='weighted')
# recall = recall_score(y_test, y_pred, average='weighted')
# print("Accuracy:", accuracy)
# print("F1 Score:", f1)
# print("Precision:", precision)
# print("Recall:", recall)
# time2 = time.time()
# print("Time taken:", time2 - time1)

# # Save the model
# from sklearn.pipeline import Pipeline
# import pickle
# Tfidf = TfidfVectorizer()
# GB = GradientBoostingClassifier()
# pipe = Pipeline([("Tfidf", Tfidf), ("GB", GB)])
# pipe.fit(X, y)

# with open('pipe.pickle', 'wb') as picklefile:
#     pickle.dump(pipe, picklefile)
