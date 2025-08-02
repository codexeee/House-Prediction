import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import sweetviz as sv
import warnings
import os

warnings.filterwarnings('ignore')

# --- Page and Plot Styling ---
st.set_page_config(
    page_title="Modern Housing Dashboard",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


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
    # Load the dataset
    df = pd.read_csv('Housing.csv')
    df_display = df.copy()

    # Convert price to PKR for both dataframes
    df['price'] = df['price'] * INR_TO_PKR_RATE
    df_display['price'] = df_display['price'] * INR_TO_PKR_RATE
    
    # Prepare the dataframe for the model (df)
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
page = st.sidebar.radio("Go to", ["Home", "Data Explorer", "Visualizations", "Comparative Analysis", "Model Insights", "Price Predictor"])

# --- Home Page ---
if page == "Home":
    st.title("🏡 Modern Housing Market Dashboard")
    st.markdown("An elegant and interactive platform to analyze and predict housing prices.")
    
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
    
    st.subheader("Descriptive Statistics")
    st.write(df_display.describe())

# --- Visualizations Page ---
elif page == "Visualizations":
    st.header("Interactive Visualizations")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df_display['price'], kde=True, ax=ax, bins=30)
        ax.set_title("Distribution of House Prices")
        st.pyplot(fig)

        st.subheader("Price vs. Area")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_display, x='area', y='price', ax=ax, alpha=0.6)
        ax.set_title("House Price vs. Area")
        st.pyplot(fig)

    with col2:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Matrix of Features")
        st.pyplot(fig)

        st.subheader("Price by Furnishing Status")
        fig, ax = plt.subplots()
        sns.boxplot(data=df_display, x='furnishingstatus', y='price', ax=ax)
        ax.set_title("Price Distribution by Furnishing Status")
        st.pyplot(fig)

# --- Comparative Analysis Page ---
elif page == "Comparative Analysis":
    st.header("Comparative Variable Analysis")
    st.markdown("Select a binary feature to split the data, and another feature to analyze within those splits.")
    
    # **FIX**: Only allow selection of binary (Yes/No) columns for the split
    binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    
    col1, col2 = st.columns(2)
    with col1:
        # This dropdown is now restricted to valid columns
        split_var = st.selectbox("Split by Feature (must be Yes/No):", binary_columns)
    with col2:
        # Allow any column to be the one analyzed
        compare_var = st.selectbox("Feature to Analyze:", df_display.columns)
        
    if st.button("Generate Comparison Report"):
        if split_var and compare_var:
            # Create the boolean condition for the split
            condition = (df_display[split_var] == 'yes')
            
            # Check if the split results in empty dataframes
            if condition.sum() == 0 or (~condition).sum() == 0:
                st.error(f"The selected feature '{split_var}' does not contain both 'yes' and 'no' values to compare. Please choose another feature.")
            else:
                # Generate the report using compare_intra
                report = sv.compare_intra(df_display, condition, [f"Has {split_var}", f"No {split_var}"], compare_var)
                
                # Save and display the report
                report_path = "comparison_report.html"
                report.show_html(report_path, open_browser=False, layout='vertical')
                
                with open(report_path, "r", encoding='utf-8') as f:
                    st.components.v1.html(f.read(), height=800, scrolling=True)
        else:
            st.warning("Please select both a feature to split by and a feature to analyze.")


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

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': np.abs(model.coef_)
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
        ax.set_title('Feature Importance in Regression Model')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Actual vs. Predicted Prices")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Price')
        ax.set_ylabel('Predicted Price')
        ax.set_title('Model Predictions vs. Actual Values')
        st.pyplot(fig)


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
            'hotwaterheating': 0,
            'airconditioning': 1,
            'parking': parking,
            'prefarea': 0,
            'furnishing_semi-furnished': 1 if furnishing == 'semi-furnished' else 0,
            'furnishing_unfurnished': 1 if furnishing == 'unfurnished' else 0,
        }
        input_df = pd.DataFrame([input_data])
        input_df = input_df[X_train.columns]
        
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        
        st.success(f"Predicted House Price: **PKR {prediction[0]:,.0f}**")
