import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_data():
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    feature_names = housing.feature_names

    df = pd.DataFrame(X, columns=feature_names)
    df["MedHouseVal"] = y
    return df, feature_names


def split_data(df, feature_names):
    X = df[feature_names].values
    y = df["MedHouseVal"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("Model Performance:")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")


def demo_prediction(model, scaler, feature_names):
    example = np.array([[8.3252, 41.0, 6.9841, 1.0238, 322.0, 2.5556, 37.88, -122.23]])

    example_scaled = scaler.transform(example)
    pred_price = model.predict(example_scaled)[0]

    print("\nExample input:")
    for name, value in zip(feature_names, example[0]):
        print(f" - {name}: {value}")

    print(f"\nPredicted median house value (x100000$): {pred_price:.3f}")


def main():
    print("Loading California Housing dataset...")
    df, feature_names = load_data()

    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    X_train, X_test, y_train, y_test = split_data(df, feature_names)
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)

    print("\nTraining Linear Regression model...")
    model = train_model(X_train_scaled, y_train)

    print("\nEvaluating model...")
    evaluate_model(model, X_test_scaled, y_test)

    print("\nDemo prediction:")
    demo_prediction(model, scaler, feature_names)


if __name__ == "__main__":
    main()
