import pickle
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import json
from sklearn.linear_model import LinearRegression


model_path = 'models/best_model.pkl'
X_test_path = 'data/processed/X_test_scaled.csv'
y_test_path = 'data/processed/y_test.csv'
y_pred_path = 'data/predictions.csv'
metrics_path = 'metrics/scores.json'


with open(model_path, 'rb') as file:
    model = pickle.load(file)

X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)


y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred)
y_pred.to_csv(y_pred_path, index=False)

metrics = {
    'r2': r2_score(y_test, y_pred),
    'mse': float(mean_squared_error(y_test, y_pred))
}

with open(metrics_path, 'w') as file:
    json.dump(metrics, file)