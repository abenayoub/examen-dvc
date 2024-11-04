import pandas as pd
from sklearn.preprocessing import MinMaxScaler

X_train = pd.read_csv('data/processed/X_train.csv', parse_dates=[0])
X_test = pd.read_csv('data/processed/X_test.csv', parse_dates=[0])

# On ssupprime les dates
X_train = X_train.drop(columns='date')
X_test = X_test.drop(columns='date')
columns = X_train.columns

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(data=X_train_scaled, columns=columns)
X_test_scaled = pd.DataFrame(data=X_test_scaled, columns=columns)

X_train_scaled.to_csv('data/processed/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('data/processed/X_test_scaled.csv', index=False)