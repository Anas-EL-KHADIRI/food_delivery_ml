import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="prediction temps de livraison",
                   page_icon="🔮",
                    layout="wide")

MODEL_PATH = "model/best_delivery_pipeline.joblib"
DATA_PATH = "data/df_clean.csv" 

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

try:
    df = load_data()
    data_loaded = True
except Exception as e:
    data_loaded = False
    data_error = str(e)

# ---------------------------------------------------------
# calculate distance
# ---------------------------------------------------------
def haversine(lat1, lng1, lat2, lng2):
    R = 6371
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

# ---------------------------------------------------------
# title + navbar
# ---------------------------------------------------------
st.title("prediction of delivery time")

tab_pred, tab_data, tab_metrics = st.tabs(
    ["prediction", "data visualisation", "model performance"]
)

# ===========================================================
# page1
# ===========================================================
with tab_pred:
    st.subheader("fill delivery informations")

    col1, col2 = st.columns(2)

    with col1:
        distance_mode = st.radio(
            "how do you want to indicate the distance?",
            ["directe distance (km)", "GPS coordinates"],
        )

        if distance_mode == "directe distance (km)":
            distance_km = st.number_input("Distance (km)", min_value=0.1, max_value=50.0, value=5.0)
        else:
            st.caption("Restaurant")
            r_lat = st.number_input("Latitude restaurant", value=12.9716, format="%.6f")
            r_lng = st.number_input("Longitude restaurant", value=77.5946, format="%.6f")
            st.caption("Point de livraison")
            d_lat = st.number_input("Latitude livraison", value=12.9800, format="%.6f")
            d_lng = st.number_input("Longitude livraison", value=77.6000, format="%.6f")
            distance_km = haversine(r_lat, r_lng, d_lat, d_lng)
            st.info(f"Distance calculée : **{distance_km:.2f} km**")

        traffic = st.selectbox("trafic density", ["Low", "Medium", "High", "Jam"])
        weather = st.selectbox(
            "meteo conditions",
            ["Sunny", "Cloudy", "Fog", "Sandstorms", "Stormy", "Windy"],
        )
        city = st.selectbox("city type", ["Urban", "Metropolitian", "Semi-Urban"])

    with col2:
        vehicle_type = st.selectbox(
            "vehicle type", ["motorcycle", "scooter", "electric_scooter", "bicycle"]
        )
        vehicle_condition = st.slider("vehicle condition (0 = bad, 3 = excellent)", 0, 3, 2)

        rating = st.slider("delivery person rating", 1.0, 5.0, 4.5, step=0.1)
        age = st.number_input("delivery person age", min_value=15, max_value=60, value=30)
        multiple_deliveries = st.selectbox("multiple deliveries", [0, 1, 2, 3])
        festival = st.selectbox("festival ?", ["No", "Yes"])
        prep_time = st.number_input(
            "preparation time (min) — delay between order and pickup",
            min_value=0,
            max_value=120,
            value=15,
        )

    st.markdown("---")

    if st.button("predict delivery time", type="primary"):
        if not model_loaded:
            st.error(f"loading model failed : {model_error}")
        else:
            input_df = pd.DataFrame(
                [
                    {
                        "Delivery_person_Age": age,
                        "multiple_deliveries": multiple_deliveries,
                        "distance_km": distance_km,
                        "prep_time_min": prep_time,
                        "Delivery_person_Ratings": rating,
                        "Vehicle_condition": vehicle_condition,
                        "Road_traffic_density": traffic,
                        "Weatherconditions": weather,
                        "Type_of_vehicle": vehicle_type,
                        "City": city,
                        "Festival": festival,
                    }
                ]
            )

            prediction = model.predict(input_df)[0]
            st.success(f"estimated delivery time : **{prediction:.1f} minutes**")

# ===========================================================
# page2
# ===========================================================
with tab_data:
    st.subheader("data visualisation")

    if not data_loaded:
        st.warning(f"Dataset not found ({DATA_PATH}). visualisation unavailable : {data_error}")
    else:
        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots()
            sns.histplot(df["Time_taken(min)"], kde=True, bins=30, ax=ax)
            ax.set_title("Distribution du temps de livraison")
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x="distance_km", y="Time_taken(min)", alpha=0.4, ax=ax)
            ax.set_title("Distance vs Temps de livraison")
            st.pyplot(fig)

        c3, c4 = st.columns(2)

        with c3:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x="Road_traffic_density", y="Time_taken(min)", ax=ax)
            ax.set_title("Temps de livraison par densité de trafic")
            st.pyplot(fig)

        with c4:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x="Weatherconditions", y="Time_taken(min)", ax=ax)
            ax.set_title("Temps de livraison par météo")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)

        st.markdown("### Aperçu des données")
        st.dataframe(df.head(50))

# ===========================================================
# page3
# ===========================================================
with tab_metrics:
    st.subheader("metrics and precision of the model ")

    if not (model_loaded and data_loaded):
        st.warning("model or/and dataset not loaded. metrics unavailable")
    else:
        feature_cols = [
            "Delivery_person_Age",
            "multiple_deliveries",
            "distance_km",
            "prep_time_min",
            "Delivery_person_Ratings",
            "Vehicle_condition",
            "Road_traffic_density",
            "Weatherconditions",
            "Type_of_vehicle",
            "City",
            "Festival",
        ]

        X = df[feature_cols]
        y = df["Time_taken(min)"]
        y_pred = model.predict(X)

        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        m1, m2, m3 = st.columns(3)
        m1.metric("MAE (min)", f"{mae:.2f}")
        m2.metric("RMSE (min)", f"{rmse:.2f}")
        m3.metric("R²", f"{r2:.3f}")

        st.markdown("### predicted vs real delivery time")
        fig, ax = plt.subplots()
        ax.scatter(y, y_pred, alpha=0.3)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--")
        ax.set_xlabel("Real Delivery Time (min)")
        ax.set_ylabel("Predicted Delivery Time (min)")
        ax.set_title("Model Accuracy")
        st.pyplot(fig)

        st.markdown("### error distribution")
        errors = y - y_pred
        fig, ax = plt.subplots()
        sns.histplot(errors, kde=True, bins=30, ax=ax)
        ax.set_title("Distribution of Residuals")
        ax.set_xlabel("Error (min)")
        st.pyplot(fig)