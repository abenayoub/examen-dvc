import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
import pickle

X_train_scaled = pd.read_csv('data/processed/X_train_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')


model = LinearRegression()
param_grid = {
    'fit_intercept': [True, False],
}

#Configuration de la recherche en grille avec validation croisée
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Lancer la GridSearchCV sur les données d'entraînement
grid_search.fit(X_train_scaled, y_train)

# Afficher les meilleurs paramètres et la meilleure performance
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score (MSE): ", -grid_search.best_score_)

best_params = grid_search.best_params_

# Sauvegarder le modèle dans un fichier .pkl
model_path = 'models/best_params.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(best_params, file)