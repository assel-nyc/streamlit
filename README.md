# 🌍 Global Life Quality and Economic Analysis Data Visualization Dashboard📊

An interactive Streamlit web application allows you to explore the relationships between poverty, life expectancy, and GDP across different countries and years.

🌎 **Deployed Interactive App**: [Click here to view the app](https://dsr5411.streamlit.app)


## Overview
This demo project showcases interactive dashboard development and data visualization using Streamlit and Plotly. It emphasizes building intuitive, responsive visualizations that enable users to explore global country data and analyze economic disparities across the world over time using real datasets. 

## Features
- **Interactive Scatter Plots**: Explore relationships between poverty, life expectancy, and GDP using interactive charts.
- **Dynamic Filtering**: Filter data by year and country to focus on specific regions or time periods.
- **Responsive Design**: The app is designed to be user-friendly and responsive, ensuring a smooth experience across devices.
- **Data Insights**: Gain insights into how economic factors affect life quality across different countries and years.      
- **ML Prediction Tool**: Random Forest model to predict life expectancy based on GDP and poverty levels, with real-time predictions displayed in the app.

Installation and setup instructions are provided in the repository to help you get started quickly.
```python
git clone https://github.com/assel-nyc/streamlit.git 
cd streamlit
pip install -r requirements.txt 
streamlit run practical.py

```
**Usage **
**Run the App and navigate through three main tabs**
- Global Overview: View yearly trends and test ML predictions.
- Country Analysis: Select a country to explore its specific data.
- Data Insights: Browse and filter the raw dataset for deeper analysis

**Dataset**
GDP per capita by country and year
Life expectancy by country and year
Poverty rate by country and year
Income inequality measures(Gini index)
Weatlth distribution metsrics (top 10% share)

**Key Learnings**
- How to build interactive web applications using Streamlit.
- Techniques for visualizing complex datasets with Plotly.
- Implementing machine learning models for real-time predictions in a web app.

**Future Enhancements**
- Adding more datasets for comprehensive analysis.
- Enhancing the ML model with additional features.
- Improving UI/UX for better user experience.
