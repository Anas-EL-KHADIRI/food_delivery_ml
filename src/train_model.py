import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


df = pd.read_csv("../data/df_clean.csv")



num_features = ["Delivery_person_Age", "distance_km", "Delivery_person_Ratings", "Vehicle_condition"]
ordinal_features = ["Road_traffic_density"]
traffic_order = [["Low", "Medium", "High", "Jam"]]
nominal_features = ["Weatherconditions", "Type_of_vehicle", "City", "Festival"]

all_features = num_features + ordinal_features + nominal_features

X = df[all_features]
y = df["Time_taken(min)"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        (
            "ord",
            OrdinalEncoder(categories=traffic_order, handle_unknown="use_encoded_value", unknown_value=-1),
            ordinal_features,
        ),
        ("nom", OneHotEncoder(drop="first", handle_unknown="ignore"), nominal_features),
    ]
)



models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Extra Trees": ExtraTreesRegressor(random_state=42, n_jobs=-1),
    "SVR": SVR(),
}

results = []

for name, model in models.items():
    print(f"Training {name}...")
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({"Model": name, "MAE (min)": mae, "RMSE (min)": rmse, "R²": r2})

results_df = pd.DataFrame(results)
print("\n--- COMPARAISON DES MODELES ---")
print(results_df)

# tuning hyperparameters for HistGradientBoostingRegressor
hist_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", HistGradientBoostingRegressor(random_state=42))]
)

param_distributions = {
    "regressor__max_iter": [100, 150, 200, 300],
    "regressor__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "regressor__max_leaf_nodes": [15, 20, 30, 40, 50],
    "regressor__min_samples_leaf": [10, 20, 30, 50],
    "regressor__l2_regularization": [0, 0.1, 1, 5, 10],
}

random_search = RandomizedSearchCV(
    estimator=hist_pipeline,
    param_distributions=param_distributions,
    n_iter=250,
    cv=3,
    scoring="neg_mean_absolute_error",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

print("\nDémarrage du tuning Hist Gradient Boosting...")
random_search.fit(X_train, y_train)
print("Tuning terminé !")

print("\n--- MEILLEURS PARAMETRES ---")
print(random_search.best_params_)
print(f"\nMeilleur MAE en cross-validation : {-random_search.best_score_:.3f} minutes")




best_hist_pipeline = random_search.best_estimator_

train_preds = best_hist_pipeline.predict(X_train)
train_mae = mean_absolute_error(y_train, train_preds)
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
train_r2 = r2_score(y_train, train_preds)

test_preds = best_hist_pipeline.predict(X_test)
test_mae = mean_absolute_error(y_test, test_preds)
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
test_r2 = r2_score(y_test, test_preds)

print("\n--- EVALUATION FINALE ---")
print(f"Train MAE:  {train_mae:.3f} min | Train RMSE: {train_rmse:.3f} min | Train R²: {train_r2:.3f}")
print(f"Test MAE:   {test_mae:.3f} min | Test RMSE:  {test_rmse:.3f} min | Test R²:  {test_r2:.3f}")

mae_gap = test_mae - train_mae
r2_gap = train_r2 - test_r2
print(f"\nMAE gap: {mae_gap:.3f} | R² gap: {r2_gap:.3f}")

n = X_test.shape[0]
p = X_test.shape[1]
test_adjusted_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)
print(f"Test Adjusted R²: {test_adjusted_r2:.3f}")



joblib.dump(best_hist_pipeline, "../models/best_delivery_pipeline.joblib")
print("\nsaved model in : models/best_delivery_pipeline.joblib")
