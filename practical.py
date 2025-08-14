import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Global Developement Explorer", layout="wide")
st.write("Welcome to your new Streamlit app.")
st.title("Worldwide Analysis of Quality of Life and Economic Factors")

#Create three tabs
tab1, tab2, tab3 = st.tabs(["Quality of Life", "Economic Factors", "Conclusion"])
with tab1:
    st.header("Quality of Life")
    st.write("This tab will contain information about the quality of life across different countries.")
    # Add more content here
with tab2:
    st.header("Economic Factors")
    st.write("This tab will contain information about the economic factors affecting different countries.")
    # Add more content here
with tab3:
    st.header("Conclusion")
    st.write("This tab will summarize the findings from the previous tabs.")
    # Add more content here


    # Load the dataset from GitHub
    data_url = "https://raw.githubusercontent.com/JohannaViktor/streamlit_practical/refs/heads/main/global_development_data.csv"
    df = pd.read_csv(data_url)

    # Show dataset
    st.write("### Full Dataset")
    st.dataframe(df)
    file_path = "https://raw.githubusercontent.com/JohannaViktor/streamlit_practical/refs/heads/main/global_development_data.csv"    
    df = pd.read_csv(file_path)
    st.write("### Full Dataset")
    st.dataframe(df)

    #Country filter 
    countries = df['country'].unique()
    Selected_countries = st.multiselect("Select countries", countries, default=countries[:5])

    min_year, max_year=int(df['year'].min()), int(df['year'].max())
    year_range = st.slider("Select year range", min_value=min_year, max_value=max_year, value=(min_year, max_year))
  
    #filter dataset
    filtered_df = df[(df['country'].isin(Selected_countries)) & (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    st.write("### Filtered Dataset")
    st.dataframe(filtered_df)

    #make dataset downloadable
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

csv_data = convert_df_to_csv(filtered_df)
st.download_button(
        
    label="Download Filtered Data",
    data=csv_data,
    file_name="filtered_global_development_data.csv")