import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.title {
    font-size: 42px;
    font-weight: bold;
    color: inherit;
    text-align: center;
}

.subtitle {
    font-size: 18px;
    color: inherit;
    text-align: center;
    margin-bottom: 30px;
}

.stButton>button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}

.result-box {
    background: linear-gradient(135deg, #2563eb, #1e3a8a);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    margin-top: 25px;
    color: white !important;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.3);
    max-width: 650px;
    margin-left: auto;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# LOAD DATA
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("AmesHousing.csv")
    return df

df = load_data()

# -----------------------------------
# DATA CLEANING
# -----------------------------------
df = df.dropna(thresh=0.7*len(df), axis=1)

num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include='object').columns
df[cat_cols] = df[cat_cols].fillna("Missing")

# -----------------------------------
# SELECT IMPORTANT FEATURES
# -----------------------------------
features = [
    'Overall Qual',
    'Gr Liv Area',
    'Garage Cars',
    'Total Bsmt SF',
    'Year Built',
    'Full Bath',
    'TotRms AbvGrd'
]

X = df[features]
y = df['SalePrice']

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# SCALING
# -----------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------
# MODEL TRAINING
# -----------------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------------
# TITLE
# -----------------------------------
st.markdown(
    '<div class="title">🏠 House Price Prediction App</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict house prices using Machine Learning & Random Forest Regression 🚀</div>',
    unsafe_allow_html=True
)

# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.header("📊 Enter House Details")

overall_qual = st.sidebar.slider(
    "Overall Quality",
    1,
    10,
    5
)

gr_liv_area = st.sidebar.slider(
    "Ground Living Area",
    500,
    5000,
    1500
)

garage_cars = st.sidebar.slider(
    "Garage Capacity",
    0,
    5,
    2
)

total_bsmt_sf = st.sidebar.slider(
    "Basement Area",
    0,
    3000,
    800
)

year_built = st.sidebar.slider(
    "Year Built",
    1900,
    2024,
    2000
)

full_bath = st.sidebar.slider(
    "Full Bathrooms",
    0,
    5,
    2
)

tot_rooms = st.sidebar.slider(
    "Total Rooms Above Ground",
    2,
    15,
    6
)

# -----------------------------------
# USER DATAFRAME
# -----------------------------------
input_data = pd.DataFrame({
    'Overall Qual': [overall_qual],
    'Gr Liv Area': [gr_liv_area],
    'Garage Cars': [garage_cars],
    'Total Bsmt SF': [total_bsmt_sf],
    'Year Built': [year_built],
    'Full Bath': [full_bath],
    'TotRms AbvGrd': [tot_rooms]
})

# -----------------------------------
# DISPLAY INPUTS
# -----------------------------------
st.subheader("📋 User Input Features")

st.dataframe(input_data)

# -----------------------------------
# PREDICTION BUTTON
# -----------------------------------
if st.button("🔮 Predict House Price"):

    prediction = model.predict(input_data)
    
    # Current approximate USD to INR rate
    usd_to_inr = 94.9

    # Convert predicted USD price to INR
    inr_price = prediction[0] * usd_to_inr

    st.markdown(f"""
    <div class="result-box">

    <h2 style="
        color:inherit;
        margin-bottom:10px;
        font-size:32px;
    ">
        🏡 Estimated House Price
    </h2>

    <h1 style="
        color:inherit;
        font-size:55px;
        margin-top:10px;
    ">
        ${prediction[0]:,.0f}
    </h1>
    
    <h3 style="
        text-align:center;
        color:white;
        margin-top:10px;
        font-size:28px;
    ">
        Approx ₹{inr_price:,.0f}
    </h3>

    </div>
    """, unsafe_allow_html=True)

# -----------------------------------
# FEATURE IMPORTANCE
# -----------------------------------
st.subheader("🔥 Feature Importance")

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

st.bar_chart(
    importance_df.set_index('Feature')
)

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("---")

st.markdown(
    "### 🚀 Built by Ashraf Shikalgar"
)

st.markdown(
    "Machine Learning Internship Project | VedGrow"
)