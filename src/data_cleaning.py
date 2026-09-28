
import numpy as np
import pandas as pd

df = pd.read_csv("../data/food_delivery_dataset.csv")
print("Shape ", df.shape)

df.drop(columns=["ID", "Delivery_person_ID"], inplace=True)

df["Time_taken(min)"] = df["Time_taken(min)"].str.replace("(min)", "", regex=False).str.strip()
df["Time_taken(min)"] = pd.to_numeric(df["Time_taken(min)"], errors="coerce")


df["Weatherconditions"] = df["Weatherconditions"].str.replace("conditions ", "", regex=False).str.strip()



cat_cols = ["Weatherconditions", "Road_traffic_density", "Type_of_order", "Type_of_vehicle", "City"]

for col in cat_cols:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace(["NaN", "nan"], np.nan)



df["Delivery_person_Age"] = pd.to_numeric(df["Delivery_person_Age"], errors="coerce")
df["Delivery_person_Ratings"] = pd.to_numeric(df["Delivery_person_Ratings"], errors="coerce")



df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d/%m/%Y", errors="coerce")
df["Time_Orderd"] = pd.to_timedelta(df["Time_Orderd"], errors="coerce")
df["Time_Order_picked"] = pd.to_timedelta(df["Time_Order_picked"], errors="coerce")



df["multiple_deliveries"] = pd.to_numeric(df["multiple_deliveries"], errors="coerce")




coord_cols = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
]
df[coord_cols] = df[coord_cols].abs()
invalid_coords_mask = (df[coord_cols] < 1.0).any(axis=1)
df = df[~invalid_coords_mask]


df["Delivery_person_Age"] = df["Delivery_person_Age"].fillna(df["Delivery_person_Age"].median())

df["Weatherconditions"] = df["Weatherconditions"].fillna(df["Weatherconditions"].mode()[0])

df["City"] = df["City"].fillna(df["City"].mode()[0])
df["Road_traffic_density"] = df["Road_traffic_density"].fillna(df["Road_traffic_density"].mode()[0])

df = df.dropna(subset=["Time_Orderd"])
df["multiple_deliveries"] = df["multiple_deliveries"].fillna(df["multiple_deliveries"].median())

df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].fillna(df["Delivery_person_Ratings"].median())



duplicate_count = df.duplicated().sum()
print("duplicates :", duplicate_count)
if duplicate_count > 0:
    df.drop_duplicates(inplace=True)

print("final shape :", df.shape)
print(df.isna().sum())



df.to_csv("../data/df_clean.csv", index=False)
print("Saved cleaned data to : data/df_clean_silver.csv")
