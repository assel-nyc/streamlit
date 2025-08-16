import streamlit as st
import pandas as pd
from io import BytesIO
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split  
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

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

# Create columns for metrics 
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Countries",
        value=len(df['country'].unique()),
        delta=None,
        help="Total number of countries in the dataset."
    )
with col2:
    st.metric(
        label="Median GDP per Capita",
        value=len(df['year'].unique()),
        delta=None,
        help="Total number of years in the dataset."
    )                   
with col3:
    st.metric(
        label="Median Life Expectancy (IHME)",
        value=df['Life Expectancy (IHME)'].median(),
        delta=None,
        help="Median life expectancy across all countries."
    )
with col4:
    st.metric(
        label="headcount_ratio_upper_mid_income_povline",
        value=df['headcount_ratio_upper_mid_income_povline'].median(),
        delta=None,
        help="Median Human Development Index (HDI) across all countries."
    )   


def scatter_life_expactancy(df, slider_value):
    filtered_df = df[(df['year'] >= slider_value[0]) & (df['year'] <= slider_value[1])]
    fig = px.scatter(
        filtered_df,
        x="GDP per capita",
        y="Life Expectancy (IHME)",
        color="country",
        size="Population",
        hover_name="country",
        log_x=True,
        size_max=60,
        title=f"GDP per Capita vs Life Expectancy (IHME) ({slider_value[0]} - {slider_value[1]})",
        labels={
            "GDP per Capita (PPP)": "GDP per Capita (USD)",
            "Life Expectancy (IHME)": "Life Expectancy (years)"
        }
    )
    fig.update_layout(transition_duration=500)
    return fig

# Call the function and display the plot (add this after the function definition)
fig = scatter_life_expactancy(filtered_df, year_range)
st.plotly_chart(fig)

st.write(df.columns)


def train_model(df):
    features = ['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year']
    target = 'Life Expectancy (IHME)'
    # Prepare the data
    X = df[['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year']]
    y = df['Life Expectancy (IHME)']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    st.write(f"### Model R2 Score: {r2:.2f}")
    st.write(f"### Model Mean Absolute Error: {mae:.2f}")
    st.write(f"### Model Mean Squared Error: {mse:.2f}")
    
    return model, features
model, features = train_model(filtered_df)  
st.write("### Model Training")
gdp_input = st.number_input("GDP per capita", min_value=float(df['GDP per capita'].min()), max_value=float(df['GDP per capita'].max()), value=float(df['GDP per capita'].median()))
poverty_input = st.number_input(
    "Headcount Ratio (Upper Mid Income Poverty Line)",
    min_value=float(df['headcount_ratio_upper_mid_income_povline'].min()),
    max_value=float(df['headcount_ratio_upper_mid_income_povline'].max()),
    value=float(df['headcount_ratio_upper_mid_income_povline'].median())
)
year_input = st.number_input("Year", min_value=int(df['year'].min()), max_value=int(df['year'].max()), value=int(df['year'].median()))

# Make predictions
if st.button("Predict Life Expectancy"):
    X_new = pd.DataFrame({
        'GDP per capita': [gdp_input],
        'headcount_ratio_upper_mid_income_povline': [poverty_input],
        'year': [year_input]
    }, columns=features)
    prediction = model.predict(X_new)[0]
    st.success(f"Predicted Life Expectancy (IHME): {prediction:.2f} years")
    st.write("### Model Features")
    importances = model.feature_importances_
    fig, ax = plt.subplots()
    ax.barh(features, importances, color='skyblue')
    ax.set_ylabel("Importance")
    st.pyplot(fig)

    