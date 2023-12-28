#Part 1

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

#Part 5

def eda():
    st.sidebar.header("Visualizations")
    
    st.header("Upload your CSV data")
    data_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if data_file is not None:
        data = pd.read_csv(data_file)
        st.write("Data Overview:")
        st.write(data.head())
        st.write(data.describe().T)
        
        plot_options = ["Bar plot", "Scatter plot", "Histogram", "Box plot"]
        selected_plot = st.sidebar.selectbox("Choose a plot type", plot_options)
        
        if selected_plot == "Bar plot":
            x_axis = st.sidebar.selectbox("Select x-axis", data.columns)
            y_axis = st.sidebar.selectbox("Select y-axis", data.columns)
            st.write("Bar plot:")
            fig, ax = plt.subplots()
            sns.barplot(x=data[x_axis], y=data[y_axis], ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")  
            st.pyplot(fig)
            
        elif selected_plot == "Scatter plot":
            x_axis = st.sidebar.selectbox("Select x-axis", data.columns)
            y_axis = st.sidebar.selectbox("Select y-axis", data.columns)
            st.write("Scatter plot:")
            fig, ax = plt.subplots()
            sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)
            st.pyplot(fig)
            
        elif selected_plot == "Histogram":
            column = st.sidebar.selectbox("Select a column", data.columns)
            bins = st.sidebar.slider("Number of bins", 5, 100, 20)
            st.write("Histogram:")
            fig, ax = plt.subplots()
            sns.histplot(data[column], bins=bins, ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Box plot":
            column = st.sidebar.selectbox("Select a column", data.columns)
            st.write("Box plot:")
            fig, ax = plt.subplots()
            sns.boxplot(data[column], ax=ax)
            st.pyplot(fig)
            

def input_data():
    bhk = st.slider(label='BHK', min_value=1, max_value=6, step=1)
    type = st.st.text_input('type', ('BHK2', 'BHK2', 'BHK3', 'BHK1'))
    locality = st.st.text_input('Furnishing Status', ('Unfurnished', 'Semi-Furnished', 'Furnished'))
    tenant = st.selectbox('Tenant Preferred', ('Bachelors/Family', 'Bachelors', 'Family'))
    bath = st.slider(label='Bathroom', min_value=1, max_value=7, step=1)
    point_c = st.selectbox('Point of Contact', ('Contact Owner', 'Contact Agent'))
    rent = st.slider(label='Rental Floor', min_value=-2, max_value=22, step=1)
    total_f = st.slider(label='Total Number of Floor', min_value=0, max_value=30, step=1)
    fixed_s = st.slider(label="Fixed Size Squere", min_value=10, max_value=3100, step=10)
    square_feet_rent = st.slider(label="Square Feet Rent", min_value=10, max_value=120, step=2)
    
    columns = [
        'BHK', 'City', 'Furnishing Status', 
        'Tenant Preferred','Bathroom', 'Point of Contact',
        'Rental Floor', 'Total Number of Floor','Fixed Size', "Square Feet Rent"
    ]
    new_data = [[bhk, city, furn_s, tenant, bath, point_c, rent, total_f, fixed_s, square_feet_rent]]
    new_data = pd.DataFrame(new_data, columns=columns)
    return new_data


def predict():
    st.write("""Predicted Rent House/Apartment/Room""")
    
    # Load the model
    with open("E:\\Guvi_Data_science\\Projects\\Smart_Predictive_Modeling_for_Rental\\decision_tree_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Collect input data using a function named input_data (you need to define this function)
    new_data = input_data()
    
    if st.button(label='Predict'):
        # Ensure that the input data is in the right format (e.g., a NumPy array)
        new_data = np.array(new_data).reshape(1, -1)
        
        # Make predictions using the loaded model
        rent_pred = model.predict(new_data)
        
        # Display the predicted rent
        st.success(f'Rent Estimate (INR): {np.round(rent_pred[0], 2)}')

        
pages = {"EDA":eda, "Predict":predict}


def main():
    st.title("Rent Price Predict")
    selected_page = st.sidebar.selectbox("Choose a page", options=list(pages.keys()))

    pages[selected_page]()


st.cache(allow_output_mutation=True)
if __name__ == "__main__":
    main()   