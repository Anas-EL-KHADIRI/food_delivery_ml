import numpy as np
import pandas as pd

df = pd.read_csv("../data/df_clean_silver.csv")

df["Time_Orderd"] = pd.to_timedelta(df["Time_Orderd"])
df["Time_Order_picked"] = pd.to_timedelta(df["Time_Order_picked"])

# calculate distance using haversine formula
R = 6371
lat1 = np.radians(df["Delivery_location_latitude"])
lng1 = np.radians(df["Delivery_location_longitude"])
lat2 = np.radians(df["Restaurant_latitude"])
lng2 = np.radians(df["Restaurant_longitude"])
dlat = lat2 - lat1
dlng = lng2 - lng1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
df["distance_km"] = 2 * R * np.arcsin(np.sqrt(a))




df["speed_kmh"] = df["distance_km"] / (df["Time_taken(min)"] / 60)
df = df[(df["speed_kmh"] >= 1) & (df["speed_kmh"] <= 95)]
print("Shape après filtrage des vitesses :", df.shape)


df["prep_time_min"] = (df["Time_Order_picked"] - df["Time_Orderd"]).dt.total_seconds() / 60.0
df["prep_time_min"] = df["prep_time_min"].apply(lambda x: x + 1440 if x < 0 else x)



df = df.drop(
    columns=[
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Time_Order_picked",
        "Time_Orderd",
    ]
)


df.to_csv("../data/df_clean.csv", index=False)
print("Fichier final sauvegardé : data/df_clean.csv")
print(df.head())
