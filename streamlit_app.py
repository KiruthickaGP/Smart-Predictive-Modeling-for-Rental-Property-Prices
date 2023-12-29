import pandas as pd
import streamlit as st
import pickle
import numpy as np
import streamlit as st
import re
from PIL import Image
from streamlit_option_menu import option_menu
import warnings
import pickle
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import pandas as pd
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

#Part 2

# Open the image file for the page icon
icon = Image.open(r"E:\Guvi_Data_science\Projects\Smart_Predictive_Modeling_for_Rental\smartrental.png")
# Set page configurations with background color
st.set_page_config(
    page_title="Smart Predictive Modeling for Rental Property Prices | By Kiruthicka",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': """# This app is created by *Kiruthicka!*"""})

# Add background color using CSS
background_color = """
<style>
    body {
        background-color: #F7EBED;  /* Set background color to #F7EBED*/            #AntiqueWhite color
    }
    .stApp {
        background-color: #F7EBED; /* Set background-color for the entire app */
    }
</style>
"""
#AntiqueWhite color #F7EBED
st.markdown(background_color, unsafe_allow_html=True)




# CREATING OPTION MENU


with st.sidebar:
    selected = option_menu(None,["Home","Predictive analysis"],
        icons=["house-fill","tools"],
        default_index=0,
        orientation="Vertical",
        styles={
            "nav-link": {
                "font-size": "30px",
                "font-family": "Fira Sans",
                "font-weight": "Bold",
                "text-align": "left",
                "margin": "10px",
                "--hover-color": "#964B00"#Brown
            },
            "icon": {"font-size": "30px"},
            "container": {"max-width": "6000px"},
            "nav-link-selected": {
                "background-color": "#CD7F32", #Bronze
                "color": "Bronze",
            }
        }
    )

#Part3
# HOME PAGE
# Open the image file for the YouTube logo
logo = Image.open(r"E:\Guvi_Data_science\Projects\Smart_Predictive_Modeling_for_Rental\smartrental.png")

# Define a custom CSS style to change text color
custom_style = """
<style>
    .black-text {
        color: black; /* Change text color to black */
    }
</style>
"""

   
# Apply the custom style
st.markdown(custom_style, unsafe_allow_html=True)

if selected == "Home":
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(r"E:\Guvi_Data_science\Projects\Smart_Predictive_Modeling_for_Rental\smartrental.png")
        st.image(image, width=700,  output_format='PNG', use_column_width=False)
        

    with col2:
        st.markdown("## :green[**Technologies Used :**]")
        st.markdown("### Python: The core programming language for data analysis, machine learning, and web application development.")
        st.markdown("### Pandas, NumPy, Matplotlib, Seaborn: Libraries for data manipulation, numerical operations, and data visualization.")
        st.markdown("### Scikit-learn: A machine learning library for building and evaluating regression and classification models.")
        st.markdown("### Streamlit: A Python library for creating interactive web applications with minimal code.")


        st.markdown("## :green[**Overview :**]")
        st.markdown("### Smart Predictive Modeling for Rental is a comprehensive project focusing on data analysis, machine learning, and web development. The project involves Python scripting for data preprocessing, exploratory data analysis (EDA), and building machine learning models for regression and classification. The Streamlit framework is used to create an interactive web page allowing users to input data and obtain predictions for selling price or lead status. ")

# Part 5
if selected == "Predictive analysis":

    
    
    # Load the data
    df = pd.read_csv("E:\\Guvi_Data_science\\Projects\\Smart_Predictive_Modeling_for_Rental\\Combined.csv")

    # Create a streamlit app
    st.title("House Rent Prediction App")

    # Display the data
    st.write(df.head())

    # Create a form to collect user input
    with st.form("user_input"):
        id = st.number_input("ID")
        type = st.selectbox("Type", df["type"].unique())
        locality = st.selectbox("Locality", df["locality"].unique())
        latitude = st.number_input("Latitude")
        longitude = st.number_input("Longitude")
        lease_type = st.selectbox("Lease Type", df["lease_type"].unique())
        gym = st.number_input("Gym")
        lift = st.number_input("Lift")
        swimming_pool = st.number_input("Swimming Pool")
        negotiable = st.number_input("Negotiable")
        furnishing = st.selectbox("Furnishing", df["furnishing"].unique())
        parking = st.number_input("Parking")
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
    if submit:
        user_input_dict = {
            "id": [id],
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
        user_input = pd.DataFrame(user_input_dict)

        # Load the model
        model = pickle.load(open("E:\\Guvi_Data_science\\Projects\\Smart_Predictive_Modeling_for_Rental\\house_rent_model.pkl", "rb"))

        # Predict the rent price
        rent_price = model.predict(user_input)

        # Display the rent price
        st.write("Predicted rent price:", rent_price[0])
