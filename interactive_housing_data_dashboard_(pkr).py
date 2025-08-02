import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# --- Page Configuration ---
st.set_page_config(
    page_title="Modern Housing Dashboard",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Conversion Rate ---
INR_TO_PKR_RATE = 3.4 

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* General Styles */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main {
        background-color: #f5f5f5;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
    }
    /* Sidebar */
    .css-1d391kg {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    .st-emotion-cache-16txtl3 {
        font-size: 20px;
        color: #2c3e50;
    }
    /* Metric Cards */
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.2s;
        margin-bottom: 20px; /* Added margin for spacing */
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-card h3 {
        font-size: 18px;
        color: #7f8c8d;
    }
    .metric-card p {
        font-size: 24px;
        font-weight: bold;
        color: #2980b9;
    }
    /* Chart Containers (Frames) */
    .chart-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    /* Buttons */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
</style>
""", unsafe_allow_html=True)


# --- Data Loading and Caching ---
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv('Housing.csv')
    df_display = df.copy()

    # Convert price to PKR
    df['price'] = df['price'] * INR_TO_PKR_RATE
    df_display['price'] = df_display['price'] * INR_TO_PKR_RATE
    
    binary_vars = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    df[binary_vars] = df[binary_vars].apply(lambda x: x.map({'yes': 1, 'no': 0}))

    furnishing_status = pd.get_dummies(df['furnishingstatus'], drop_first=True, prefix='furnishing')
    df = pd.concat([df, furnishing_status], axis=1)
    df.drop(['furnishingstatus'], axis=1, inplace=True)
    
    return df, df_display

df, df_display = load_and_prep_data()

# --- Model Training ---
@st.cache_resource
def train_models(df):
    features = df.columns.drop('price')
    X = df[features]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = MinMaxScaler()
    X_train[features] = scaler.fit_transform(X_train[features])
    X_test[features] = scaler.transform(X_test[features])
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test, scaler

model, X_train, X_test, y_train, y_test, scaler = train_models(df)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Explorer", "Visualizations", "Model Insights", "Price Predictor"])

# --- Home Page ---
if page == "Home":
    st.title("🏡 Modern Housing Market Dashboard")
    st.markdown("An elegant and interactive platform to analyze and predict housing prices.")
    
    # Hero Image
    st.image("https://images.pexels.com/photos/106399/pexels-photo-106399.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1", use_column_width=True)
    
    st.header("Key Metrics at a Glance")
    
    avg_price = df_display['price'].mean()
    avg_area = df_display['area'].mean()
    num_houses = len(df_display)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>Average Price (PKR)</h3><p>{avg_price:,.0f}</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>Average Area (sq. ft.)</h3><p>{avg_area:,.0f}</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h3>Total Properties</h3><p>{num_houses}</p></div>', unsafe_allow_html=True)

# --- Data Explorer Page ---
elif page == "Data Explorer":
    st.header("Data Explorer")
    st.markdown("A detailed look into the raw dataset.")
    st.dataframe(df_display, height=500)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Info")
        st.text(f"Dataset Shape: {df.shape}")
        st.text(f"Missing Values: {df_display.isnull().sum().sum()}")
    with col2:
        st.subheader("Data Types")
        st.dataframe(df_display.dtypes.astype(str), use_container_width=True)

# --- Visualizations Page ---
elif page == "Visualizations":
    st.header("Interactive Visualizations")
    
    with st.container():
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Price vs. Area")
        fig_scatter = px.scatter(df_display, x='area', y='price', color='airconditioning',
                                 hover_data=['bedrooms', 'bathrooms'],
                                 title='Price vs. Area with Air Conditioning',
                                 labels={'price': 'Price (PKR)', 'area': 'Area (sq. ft.)'})
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Categorical Features vs. Price")
        qualitative_vars = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                           'airconditioning', 'prefarea', 'furnishingstatus']
        selected_feature = st.selectbox("Select a feature:", qualitative_vars)
        fig_box = px.box(df_display, x=selected_feature, y='price', color=selected_feature,
                         title=f'Price Distribution by {selected_feature.title()}',
                         labels={'price': 'Price (PKR)'})
        st.plotly_chart(fig_box, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- Model Insights Page ---
elif page == "Model Insights":
    st.header("Regression Model Insights")
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>R² Score</h3><p>{r2:.4f}</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>RMSE (PKR)</h3><p>{rmse:,.0f}</p></div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': np.abs(model.coef_)
        }).sort_values('importance', ascending=False)
        fig_importance = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                                title='Feature Importance in Regression Model')
        st.plotly_chart(fig_importance, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- Price Predictor Page ---
elif page == "Price Predictor":
    st.header("House Price Predictor")
    
    with st.form("prediction_form"):
        st.markdown("Enter the details of the house to get a price prediction.")
        col1, col2, col3 = st.columns(3)
        with col1:
            area = st.number_input("Area (sq. ft.)", min_value=1000, max_value=20000, value=3500, step=100)
            bedrooms = st.slider("Bedrooms", 1, 6, 3)
            bathrooms = st.slider("Bathrooms", 1, 4, 2)
        with col2:
            stories = st.slider("Stories", 1, 4, 2)
            parking = st.slider("Parking Spaces", 0, 4, 1)
            mainroad = st.radio("Main Road Access", ["Yes", "No"])
        with col3:
            guestroom = st.radio("Guest Room", ["Yes", "No"])
            basement = st.radio("Basement", ["Yes", "No"])
            furnishing = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

        submit_button = st.form_submit_button(label='Predict Price')

    if submit_button:
        input_data = {
            'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms, 'stories': stories,
            'mainroad': 1 if mainroad == 'Yes' else 0,
            'guestroom': 1 if guestroom == 'Yes' else 0,
            'basement': 1 if basement == 'Yes' else 0,
            'hotwaterheating': 0, # Assuming no hot water by default
            'airconditioning': 1, # Assuming AC by default
            'parking': parking,
            'prefarea': 0, # Assuming not preferred area by default
            'furnishing_semi-furnished': 1 if furnishing == 'semi-furnished' else 0,
            'furnishing_unfurnished': 1 if furnishing == 'unfurnished' else 0,
        }
        input_df = pd.DataFrame([input_data])
        input_df = input_df[X_train.columns]
        
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        
        st.success(f"Predicted House Price: **PKR {prediction[0]:,.0f}**")
