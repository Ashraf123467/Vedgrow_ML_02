import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import streamlit.components.v1 as components
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

tot_rms_abv_grd = st.sidebar.slider(
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
    'tot_rms_abv_grd': [tot_rms_abv_grd]
})

# -----------------------------------
# DISPLAY INPUTS
# -----------------------------------
st.subheader("📋 User Input Features")

st.dataframe(input_data)

# -----------------------------------
# PREDICTION BUTTON
# -----------------------------------
if st.button("🪄 Predict House Price"):

    # Input Features

    features = np.array([[
        overall_qual,
        gr_liv_area,
        garage_cars,
        total_bsmt_sf,
        year_built,
        full_bath,
        tot_rms_abv_grd
    ]])

    # Prediction

    predicted_price = model.predict(features)[0]

    # USD to INR Conversion

    usd_to_inr = 94.9

    inr_price = predicted_price * usd_to_inr


    # Indian Currency Formatter

    def format_indian_currency(number):

        number = int(number)

        s = str(number)

        last_three = s[-3:]

        remaining = s[:-3]

        if remaining != "":

            remaining = ",".join(
                [
                    remaining[max(i-2,0):i]

                    for i in range(
                        len(remaining),
                        0,
                        -2
                    )
                ][::-1]
            )

            return remaining + "," + last_three

        else:

            return last_three


    formatted_inr = format_indian_currency(
        inr_price
    )


    # Display Values

    usd_price = f"${predicted_price:,.0f}"

    inr_display = f"Approx ₹{formatted_inr}"


    # Premium Result Card

    components.html(
    f"""

    <div style="
        background: linear-gradient(135deg,#2563eb,#1d4ed8);
        padding:45px;
        border-radius:30px;
        text-align:center;
        margin-top:35px;
        box-shadow:0px 12px 30px rgba(0,0,0,0.35);
        color:white;
        font-family:Poppins,sans-serif;
        width:85%;
        margin-left:auto;
        margin-right:auto;
    ">

        <div style="
            font-size:28px;
            font-weight:600;
            margin-bottom:25px;
            opacity:0.95;
        ">
            🏡 Estimated House Price
        </div>

        <div style="
            font-size:72px;
            font-weight:800;
            margin-bottom:20px;
            letter-spacing:1px;
        ">
            {usd_price}
        </div>

        <div style="
            width:120px;
            height:4px;
            background:white;
            margin:0 auto 25px auto;
            border-radius:10px;
            opacity:0.7;
        ">
        </div>

        <div style="
            font-size:32px;
            font-weight:600;
            opacity:0.95;
        ">
            {inr_display}
        </div>

    </div>

    """,

    height=420
)
# -----------------------------------
# FEATURE IMPORTANCE
# -----------------------------------

st.markdown("## 🔥 Feature Importance")

feature_names = [

    "Overall Quality",

    "Living Area",

    "Garage Cars",

    "Basement Area",

    "Year Built",

    "Full Bathrooms",

    "Total Rooms"

]

importance_df = pd.DataFrame({

    'Feature': feature_names,

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
