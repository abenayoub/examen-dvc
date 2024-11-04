import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

X_train = pd.read_csv('data/processed/X_train_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
best_params_path = 'models/best_params.pkl'
model_path = 'models/lr_model.pkl'

with open(best_params_path, 'rb') as file:
    params = pickle.load(file)

model = LinearRegression(**params)
model.fit(X_train, y_train)

with open(model_path, 'wb') as file:
    pickle.dump(model, file)