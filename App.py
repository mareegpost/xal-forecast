import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


st.title("🐐 Goat Market Price Predictor")

st.write("Upload goat dataset and predict goat market price using Machine Learning.")


# Upload Dataset
uploaded_file = st.file_uploader("Upload goat_prices.csv", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    st.write(f"Dataset Shape: {df.shape}")


    # Data Exploration
    st.subheader("Price Statistics")
    st.write(df["price_usd"].describe())


    # Scatter Plot
    st.subheader("Weight vs Price")

    fig, ax = plt.subplots()
    ax.scatter(df["weight_kg"], df["price_usd"])
    ax.set_xlabel("Weight (kg)")
    ax.set_ylabel("Price (USD)")
    ax.set_title("Live Weight vs Market Price")

    st.pyplot(fig)


    # Prepare Data
    X = df[['age_months', 'weight_kg', 'health_score']]
    y = df['price_usd']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)


    st.subheader("Model Insights")

    st.write(f"Weight importance: {model.coef_[1]:.2f} USD per kg")


    # Evaluate
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    st.write(f"RMSE: ${rmse:.2f}")
    st.write(f"R² Score: {r2:.2%}")


    # Prediction Section
    st.subheader("Predict Goat Price")

    age = st.number_input("Age (months)", 1, 60, 10)
    weight = st.number_input("Weight (kg)", 10, 100, 35)
    health = st.slider("Health Score", 1, 10, 8)

    if st.button("Predict Price"):

        new_goat = np.array([[age, weight, health]])

        predicted_price = model.predict(new_goat)[0]

        st.success(f"Predicted Goat Price: ${predicted_price:.2f}")
