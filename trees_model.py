import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Load data
df = pd.read_csv("../data/tree_data.csv")

# Features and target
X = df[['population', 'existing_trees', 'area_sq_km', 'aqi']]
y = df['trees_needed']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Save model
pickle.dump(model, open("../models/tree_data_model.pkl", "wb"))
