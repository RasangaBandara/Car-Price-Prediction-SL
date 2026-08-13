import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title='Used Car Price Predictor (SL Market Sync)', page_icon='🚗', layout='centered'
)

st.title('🚗 Used Car Price Prediction System')
st.subheader('Sri Lankan Market Context-Aware Model')
st.markdown('Input the vehicle specifications below to estimate its market value.')
st.markdown('---')

# --- Load Data & Train Model (Cached for Performance) ---
@st.cache_data
def load_and_train_model():
    df = pd.read_csv('car_price_dataset.csv')
    df_clean = df.drop(columns=['Unnamed: 0', 'Date'], errors='ignore')
    
    # Standardize text columns to uppercase
    df_clean['Brand'] = df_clean['Brand'].astype(str).str.upper().str.strip()
    df_clean['Model'] = df_clean['Model'].astype(str).str.upper().str.strip()
    df_clean['Town'] = df_clean['Town'].astype(str).str.upper().str.strip()
    
    # -----------------------------------------------------------------
    # DATA SCIENCE FIX 1: OUTLIER REMOVAL (ඉතා ඉහළ සහ ඉතා අඩු වැරදි මිල ගණන් ඉවත් කිරීම)
    # Removing extreme anomalies that distort the Random Forest learning process
    q_low = df_clean["Price"].quantile(0.01)
    q_hi  = df_clean["Price"].quantile(0.99)
    df_clean = df_clean[(df_clean["Price"] < q_hi) & (df_clean["Price"] > q_low)]
    # -----------------------------------------------------------------

    dataset_max_year = int(df_clean['YOM'].max())

    # Clean binary flags
    bool_cols = ['AIR CONDITION', 'POWER STEERING', 'POWER MIRROR', 'POWER WINDOW']
    for col in bool_cols:
        if col in df_clean.columns:
            df_clean[col] = (df_clean[col] == 'Available').astype(int)

    if 'Leasing' in df_clean.columns:
        df_clean['Leasing'] = (df_clean['Leasing'] != 'No Leasing').astype(int)
    if 'Condition' in df_clean.columns:
        df_clean['Condition'] = (df_clean['Condition'] == 'USED').astype(int)

    # Core Domain Features
    df_clean['Car_Age'] = dataset_max_year - df_clean['YOM']
    df_clean['Mileage_Per_Year'] = df_clean['Millage(KM)'] / (df_clean['Car_Age'] + 1)
    df_clean['Is_Import_Ban_Period'] = (df_clean['YOM'] >= 2020).astype(int)

    # High-Fidelity Target Encoding
    brand_target_map = df_clean.groupby('Brand')['Price'].mean().to_dict()
    model_target_map = df_clean.groupby('Model')['Price'].mean().to_dict()
    town_target_map = df_clean.groupby('Town')['Price'].mean().to_dict()

    df_clean['Brand_encoded'] = df_clean['Brand'].map(brand_target_map)
    df_clean['Model_encoded'] = df_clean['Model'].map(model_target_map)
    df_clean['Town_encoded'] = df_clean['Town'].map(town_target_map)

    # Max price boundary tracking per model to enforce hard caps later
    model_max_market_cap = df_clean.groupby('Model')['Price'].max().to_dict()

    unique_brands = sorted(df_clean['Brand'].unique().tolist())
    unique_towns = sorted(df_clean['Town'].unique().tolist())
    raw_brand_model_df = df_clean[['Brand', 'Model']].drop_duplicates()

    df_encoded = pd.get_dummies(df_clean, columns=['Gear', 'Fuel Type'], drop_first=True)
    df_encoded = df_encoded.drop(columns=['Brand', 'Model', 'Town'])

    X = df_encoded.drop(columns=['Price'])
    y = df_encoded['Price']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, X.columns, brand_target_map, model_target_map, town_target_map, unique_brands, unique_towns, raw_brand_model_df, dataset_max_year, model_max_market_cap

# Initialize configurations
model, feature_columns, brand_map, model_map, town_map, brands, towns, brand_model_df, max_year, max_caps = load_and_train_model()

# --- UI Input Form Layout ---
col1, col2 = st.columns(2)

with col1:
    selected_brand = st.selectbox('Select Vehicle Brand', options=brands, format_func=lambda x: str(x).title())
    selected_yom = st.slider('Year of Manufacture (YOM)', min_value=1980, max_value=max_year, value=2015, step=1)
    selected_gear = st.selectbox('Transmission Type', options=['Automatic', 'Manual'])
    selected_fuel = st.selectbox('Fuel Type', options=['Petrol', 'Diesel', 'Hybrid'])

with col2:
    filtered_models = sorted(brand_model_df[brand_model_df['Brand'] == selected_brand]['Model'].tolist())
    selected_model = st.selectbox('Select Vehicle Model', options=filtered_models if filtered_models else [''], format_func=lambda x: str(x).title())
    selected_mileage = st.number_input('Total Mileage (KM)', min_value=0, max_value=1000000, value=149000, step=1000)
    selected_engine = st.number_input('Engine Capacity (cc)', min_value=500, max_value=5000, value=1500, step=100)
    
    default_town_index = towns.index('COLOMBO') if 'COLOMBO' in towns else 0
    selected_town = st.selectbox('Market Location (Town)', options=towns, index=default_town_index, format_func=lambda x: str(x).title())

st.markdown('### Additional Equipment / Options')
c1, c2, c3, c4 = st.columns(4)
ac = c1.checkbox('Air Conditioning', value=True)
ps = c2.checkbox('Power Steering', value=True)
pm = c3.checkbox('Power Mirrors', value=True)
pw = c4.checkbox('Power Windows', value=True)

# --- Inference and Output ---
st.markdown('---')
if st.button('Estimate Market Price', type='primary'):
    car_age = max_year - selected_yom
    mileage_per_year = float(selected_mileage) / float(car_age + 1)
    is_import_ban = 1.0 if selected_yom >= 2020 else 0.0

    b_lookup = str(selected_brand).upper().strip()
    m_lookup = str(selected_model).upper().strip()
    t_lookup = str(selected_town).upper().strip()

    b_val = float(brand_map.get(b_lookup, np.median(list(brand_map.values()))))
    m_val = float(model_map.get(m_lookup, np.median(list(model_map.values()))))
    t_val = float(town_map.get(t_lookup, np.median(list(town_map.values()))))

    # Compile dynamic input vector
    input_row = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    input_row.loc[0, 'YOM'] = float(selected_yom)
    input_row.loc[0, 'Engine (cc)'] = float(selected_engine)
    input_row.loc[0, 'Millage(KM)'] = float(selected_mileage)

    if 'AIR CONDITION' in feature_columns: input_row.loc[0, 'AIR CONDITION'] = float(ac)
    if 'POWER STEERING' in feature_columns: input_row.loc[0, 'POWER STEERING'] = float(ps)
    if 'POWER MIRROR' in feature_columns: input_row.loc[0, 'POWER MIRROR'] = float(pm)
    if 'POWER WINDOW' in feature_columns: input_row.loc[0, 'POWER WINDOW'] = float(pw)
    if 'Leasing' in feature_columns: input_row.loc[0, 'Leasing'] = 0.0
    if 'Condition' in feature_columns: input_row.loc[0, 'Condition'] = 1.0

    input_row.loc[0, 'Car_Age'] = float(car_age)
    input_row.loc[0, 'Mileage_Per_Year'] = float(mileage_per_year)
    input_row.loc[0, 'Is_Import_Ban_Period'] = is_import_ban
    
    input_row.loc[0, 'Brand_encoded'] = b_val
    input_row.loc[0, 'Model_encoded'] = m_val
    input_row.loc[0, 'Town_encoded'] = t_val

    if selected_gear == 'Manual' and 'Gear_Manual' in feature_columns:
        input_row.loc[0, 'Gear_Manual'] = 1.0
    if selected_fuel == 'Diesel' and 'Fuel Type_Diesel' in feature_columns:
        input_row.loc[0, 'Fuel Type_Diesel'] = 1.0
    elif selected_fuel == 'Hybrid' and 'Fuel Type_Hybrid' in feature_columns:
        input_row.loc[0, 'Fuel Type_Hybrid'] = 1.0

    # Base Model Prediction
    prediction = model.predict(input_row)[0]

    # -----------------------------------------------------------------
    # DATA SCIENCE FIX 2: MARKET CALIBRATION CAPS (මිල අසාමාන්‍ය ලෙස ඉහළ යාම වැළැක්වීම)
    # Applying realistic mileage based depreciation caps to align with Ikman.lk realities
    model_max = max_caps.get(m_lookup, prediction)
    
    # 1. High Mileage Depreciation Penalty
    if selected_mileage > 120000:
        prediction *= 0.92  # 8% drop for high wear and tear
        
    # 2. Hard Ceiling Check: Make sure it doesn't absurdly overshoot the general vehicle class cap
    if prediction > model_max:
        prediction = model_max * 0.95
    # -----------------------------------------------------------------

    st.success('Prediction Successful!')
    st.metric(
        label="Estimated Value",
        value=f"{prediction:.2f} Lakhs LKR",
        delta="Outlier-Sanitized Market Match"
    )