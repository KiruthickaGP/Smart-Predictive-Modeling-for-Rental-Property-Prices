# prompt: build a streamlit application to predict rent price  add all the feature names  0   id               20377 non-null  int16    1   type             20377 non-null  int8     2   locality         20377 non-null  int16    3   activation_date  20377 non-null  int16    4   latitude         20377 non-null  float64  5   longitude        20377 non-null  float64  6   lease_type       20377 non-null  int8     7   gym              20377 non-null  float64  8   lift             20377 non-null  float64  9   swimming_pool    20377 non-null  float64  10  negotiable       20377 non-null  float64  11  furnishing       20377 non-null  int8     12  parking          20377 non-null  int8     13  property_size    20377 non-null  float64  14  property_age     20377 non-null  float64  15  bathroom         20377 non-null  float64  16  facing           20377 non-null  int8     17  cup_board        20377 non-null  float64  18  floor            20377 non-null  float64  19  total_floor      20377 non-null  float64  20  amenities        20377 non-null  int16    21  water_supply     20377 non-null  int8     22  building_type    20377 non-null  int8     23  balconies        20377 non-null  float64  24  rent

from turtle import st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
from PIL import Image
import streamlit as st

# Load the data
df = pd.read_csv(r"E:\Guvi_Data_science\Projects\Smart_Predictive_Modeling_for_Rental\Combined.csv")

# Create a streamlit app
st.title("House Rent Prediction App")

# Display the data
st.write(df.head())

# Create a form to collect user input
with st.form("user_input"):
    type = st.selectbox("Type", df["type"].unique())
    locality = st.selectbox("Locality", df["locality"].unique())
    latitude = st.number_input("Latitude")
    longitude = st.number_input("Longitude")
    lease_type = st.selectbox("Lease Type", df["lease_type"].unique())
    gym = st.selectbox("Gym", df["gym"].unique())
    lift = st.selectbox("Lift", df["lift"].unique())
    swimming_pool = st.selectbox("Swimming Pool", df["swimming_pool"].unique())
    negotiable = st.selectbox("Negotiable", df["negotiable"].unique())
    furnishing = st.selectbox("Furnishing", df["furnishing"].unique())
    parking = st.selectbox("Parking", df["parking"].unique())
    property_size = st.number_input("Property Size")
    property_age = st.number_input("Property Age")
    bathroom = st.number_input("Bathroom")
    facing = st.selectbox("Facing", df["facing"].unique())
    cup_board = st.number_input("Cup Board")
    floor = st.number_input("Floor")
    total_floor = st.number_input("Total Floor")
    amenities = st.selectbox("Amenities", df["amenities"].unique())
    water_supply = st.selectbox("Water Supply", df["water_supply"].unique())
    building_type = st.selectbox("Building Type", df["building_type"].unique())
    balconies = st.number_input("Balconies")

    # Submit the form
    submit = st.form_submit_button("Predict")

# Predict the rent price
    user_input_dict = {
        "type": [type],
        "locality": [locality],
        "latitude": [latitude],
        "longitude": [longitude],
        "lease_type": [lease_type],
        "gym": [gym],
        "lift": [lift],
        "swimming_pool": [swimming_pool],
        "negotiable": [negotiable],
        "furnishing": [furnishing],
        "parking": [parking],
        "property_size": [property_size],
        "property_age": [property_age],
        "bathroom": [bathroom],
        "facing": [facing],
        "cup_board": [cup_board],
        "floor": [floor],
        "total_floor": [total_floor],
        "amenities": [amenities],
        "water_supply": [water_supply],
        "building_type": [building_type],
        "balconies": [balconies]
    }

    # Create a DataFrame with the user input and set the index
    user_input = pd.DataFrame(user_input_dict, index=[0])

    # Load the model
    model = pickle.load(open(r"E:\Guvi_Data_science\Projects\Smart_Predictive_Modeling_for_Rental\linear_regression_model.pkl", "rb"))

    # Predict the rent price
    rent_price = model.predict(user_input)

    # Display the rent price
    st.write("Predicted rent price:", rent_price)