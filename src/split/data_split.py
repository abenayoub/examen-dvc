import pandas as pd
import requests
from sklearn.model_selection import train_test_split

output_path = 'data/raw/raw.csv'

r = requests.get("https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv")
with open(output_path, "wb") as file:
    file.write(r.content)

df = pd.read_csv('data/raw/raw.csv')
X = df.drop(columns='silica_concentrate')
y = df['silica_concentrate']




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)