import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from scipy import stats
import warnings

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
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)
    
    # Scikit-learn model for prediction
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Statsmodels for detailed analysis and confidence intervals
    X_train_sm = sm.add_constant(X_train_scaled)
    sm_model = sm.OLS(y_train.values, X_train_sm).fit()
    
    return model, X_train_scaled, X_test_scaled, y_train, y_test, scaler, sm_model

model, X_train, X_test, y_train, y_test, scaler, sm_model = train_models(df)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Explorer", "Distribution Analysis", "Categorical Analysis", "Comparative Analysis", "Model Insights", "Price Predictor"])

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
    
    st.subheader("Confidence Intervals for Mean (95%)")
    ci_data = []
    numerical_cols = df_display.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        data = df_display[col].dropna()
        if len(data) > 1:
            ci = stats.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=stats.sem(data))
            ci_data.append({'Variable': col, 'Lower Bound': ci[0], 'Upper Bound': ci[1]})
    ci_df = pd.DataFrame(ci_data)
    st.dataframe(ci_df)

# --- Distribution Analysis Page ---
elif page == "Distribution Analysis":
    st.header("Comprehensive Distribution Analysis")
    st.markdown("Select a numerical variable to see its distribution against the price.")
    
    numerical_columns = df_display.select_dtypes(include=np.number).columns.tolist()
    # Exclude price from the list of variables to select, as it's the target
    variable_to_plot = st.selectbox("Select a variable to analyze:", [col for col in numerical_columns if col != 'price'])
    
    if variable_to_plot:
        st.subheader(f"Joint Distribution of Price and {variable_to_plot.title()}")
        
        # Create the jointplot
        g = sns.jointplot(data=df_display, x=variable_to_plot, y="price", kind="reg", 
                          joint_kws={'line_kws':{'color':'red'}})
        st.pyplot(g)
        
        st.subheader(f"Statistical Summary for {variable_to_plot.title()}")
        summary = stats.describe(df_display[variable_to_plot])
        st.text(summary)


# --- Categorical Analysis Page ---
elif page == "Categorical Analysis":
    st.header("Categorical Variable Analysis")
    st.markdown("Select a categorical variable to explore its distribution and impact on price.")
    
    categorical_columns = df_display.select_dtypes(include=['object']).columns.tolist()
    
    selected_cat_var = st.selectbox("Select a Categorical Variable:", categorical_columns)
    
    if selected_cat_var:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Frequency of {selected_cat_var}")
            fig, ax = plt.subplots()
            sns.countplot(y=df_display[selected_cat_var], ax=ax, order = df_display[selected_cat_var].value_counts().index)
            ax.set_title(f"Distribution of {selected_cat_var}")
            st.pyplot(fig)

        with col2:
            st.subheader(f"Price vs. {selected_cat_var}")
            fig, ax = plt.subplots()
            sns.boxplot(x=df_display[selected_cat_var], y=df_display['price'], ax=ax)
            ax.set_title(f"Price Distribution by {selected_cat_var}")
            plt.xticks(rotation=45)
            st.pyplot(fig)


# --- Comparative Analysis Page ---
elif page == "Comparative Analysis":
    st.header("Comparative Variable Analysis")
    st.markdown("Select a binary feature to split the data, and a numerical feature to analyze within those splits.")
    
    binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    numerical_columns = df_display.select_dtypes(include=np.number).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        split_var = st.selectbox("Split by Feature (Yes/No):", binary_columns)
    with col2:
        compare_var = st.selectbox("Feature to Analyze (Numerical):", numerical_columns)
        
    if st.button("Generate Comparison"):
        if split_var and compare_var:
            df_yes = df_display[df_display[split_var] == 'yes']
            df_no = df_display[df_display[split_var] == 'no']

            if df_yes.empty or df_no.empty:
                st.error(f"The selected feature '{split_var}' does not contain both 'yes' and 'no' values to compare. Please choose another feature.")
            else:
                st.subheader(f"Comparison of '{compare_var}' based on '{split_var}'")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Group: Has {split_var}**")
                    st.write(df_yes[[compare_var]].describe())
                
                with col2:
                    st.markdown(f"**Group: No {split_var}**")
                    st.write(df_no[[compare_var]].describe())
                
                st.subheader("Distribution Plot")
                fig, ax = plt.subplots()
                sns.kdeplot(df_yes[compare_var], ax=ax, fill=True, label=f'Has {split_var}')
                sns.kdeplot(df_no[compare_var], ax=ax, fill=True, label=f'No {split_var}')
                ax.legend()
                ax.set_title(f"Distribution of {compare_var} by {split_var}")
                st.pyplot(fig)

                st.subheader("Box Plot")
                fig, ax = plt.subplots()
                sns.boxplot(x=df_display[split_var], y=df_display[compare_var], ax=ax)
                ax.set_title(f"Box Plot of {compare_var} by {split_var}")
                st.pyplot(fig)
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

    st.subheader("Regression Coefficients & Confidence Intervals")
    conf_int = sm_model.conf_int()
    conf_int.columns = ['Lower CI', 'Upper CI']
    params_df = pd.DataFrame({'Coefficient': sm_model.params, 'Std.Err': sm_model.bse, 'P-value': sm_model.pvalues})
    params_df = params_df.join(conf_int)
    st.dataframe(params_df.style.format("{:.4f}"))

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
        # Create a dataframe for the input
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
        input_df = pd.DataFrame([input_data], columns=X_train.columns)
        
        # Scale the input data
        input_scaled = scaler.transform(input_df)
        
        # Predict the price
        prediction = model.predict(input_scaled)
        
        st.success(f"Predicted House Price: **PKR {prediction[0]:,.0f}**")
