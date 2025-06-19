import streamlit as st
import pandas as pd
import joblib
import datetime

# Load trained model and encoders
model = joblib.load("rainfalllightgbm_model.pkl")
label_encoder_location = joblib.load("label_encoder_location.pkl")
label_encoder_windgust = joblib.load("label_encoder_windgust.pkl")
label_encoder_wind9am = joblib.load("label_encoder_wind9am.pkl")
label_encoder_wind3pm = joblib.load("label_encoder_wind3pm.pkl")

# Session state for prediction history
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "MinTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed",
        "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
        "Pressure9am", "Cloud9am", "Cloud3pm", "Location", "WindGustDir",
        "WindDir9am", "WindDir3pm", "RainToday", "year", "month", "day", "Prediction"
    ])
import streamlit as st
import pandas as pd
import streamlit as st

# Location list from your dataset
location_names = [
    "Adelaide", "Albany", "Albury", "AliceSprings", "BadgerysCreek", "Ballarat", "Bendigo", "Brisbane", "Cairns", "Canberra",
    "Cobar", "CoffsHarbour", "Dartmoor", "Darwin", "GoldCoast", "Hobart", "Katherine", "Launceston", "Melbourne", "MelbourneAirport",
    "Mildura", "Moree", "MountGambier", "MountGinini", "Newcastle", "Nhil", "NorahHead", "NorfolkIsland", "Nuriootpa", "PearceRAAF",
    "Penrith", "Perth", "PerthAirport", "Portland", "Richmond", "Sale", "SalmonGums", "Sydney", "SydneyAirport", "Townsville",
    "Tuggeranong", "Uluru", "WaggaWagga", "Walpole", "Watsonia", "Williamtown", "Witchcliffe", "Wollongong", "Woomera"
]
# Create mapping DataFrame
location_df = pd.DataFrame({
    "Location Name": location_names,
    "Encoded Value": list(range(len(location_names)))
})
# Display as table inside expander
with st.expander("📍 Location Encoding Reference (Click to Expand)", expanded=False):
    st.dataframe(location_df.style.format({"Encoded Value": "{:d}"}), use_container_width=True)
# Wind directions
wind_directions = [
    "E", "ENE", "ESE", "N", "NE", "NNE", "NNW", "NW",
    "S", "SE", "SSE", "SSW", "SW", "W", "WNW", "WSW"
]
wind_df = pd.DataFrame({
    "Direction": wind_directions,
    "Encoded Value": list(range(len(wind_directions)))
})
with st.expander("🧭 Wind Direction Encoding Reference (Click to Expand)", expanded=False):
    st.dataframe(wind_df.style.format({"Encoded Value": "{:d}"}), use_container_width=True)

st.title("🌧 Rain Tomorrow Prediction App")
st.write("Fill in today's weather details to predict if it will rain tomorrow.")
# Sidebar
st.sidebar.header("🌦 Enter Weather Features")
# === Numerical Inputs ===
MinTemp = st.sidebar.number_input("Min Temperature (°C)", -10.0, 50.0, 15.0)
Rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 200.0, 5.0)
Evaporation = st.sidebar.number_input("Evaporation (mm)", 0.0, 50.0, 5.0)
Sunshine = st.sidebar.number_input("Sunshine (hours)", 0.0, 24.0, 6.0)
WindGustSpeed = st.sidebar.number_input("Wind Gust Speed (km/h)", 0.0, 200.0, 40.0)
WindSpeed9am = st.sidebar.number_input("Wind Speed at 9AM (km/h)", 0.0, 100.0, 10.0)
WindSpeed3pm = st.sidebar.number_input("Wind Speed at 3PM (km/h)", 0.0, 100.0, 20.0)
Humidity9am = st.sidebar.slider("Humidity at 9AM (%)", 0, 100, 70)
Humidity3pm = st.sidebar.slider("Humidity at 3PM (%)", 0, 100, 50)
Pressure9am = st.sidebar.number_input("Pressure at 9AM (hPa)", 900.0, 1100.0, 1010.0)
Cloud9am = st.sidebar.slider("Cloud at 9AM (oktas)", 0, 8, 4)
Cloud3pm = st.sidebar.slider("Cloud at 3PM (oktas)", 0, 8, 4)

# === Dropdowns for Categorical Inputs (with visible encoding for user clarity) ===

# Location
location_options = list(label_encoder_location.classes_)
location_name = st.sidebar.selectbox(
    "📍 Location (will be encoded automatically)", 
    location_options, 
    help="Choose the city name; it will be internally converted to a numerical value"
)
Location = label_encoder_location.transform([location_name])[0]
st.sidebar.markdown(f"**Selected Location Encoded As:** `{Location}`")

# Wind Gust Direction
gust_options = list(label_encoder_windgust.classes_)
gust_name = st.sidebar.selectbox(
    "🌬️ Wind Gust Direction", 
    gust_options, 
    help="E.g., N = North, S = South, NW = Northwest"
)
WindGustDir = label_encoder_windgust.transform([gust_name])[0]
st.sidebar.markdown(f"**Encoded Gust Direction:** `{WindGustDir}`")

# Wind Direction at 9AM
wind9am_options = list(label_encoder_wind9am.classes_)
wind9am_name = st.sidebar.selectbox(
    "🧭 Wind Direction at 9AM", 
    wind9am_options,
    help="Select cardinal wind direction at 9AM (e.g., E, SE, W)"
)
WindDir9am = label_encoder_wind9am.transform([wind9am_name])[0]
st.sidebar.markdown(f"**Encoded WindDir9am:** `{WindDir9am}`")

# Wind Direction at 3PM
wind3pm_options = list(label_encoder_wind3pm.classes_)
wind3pm_name = st.sidebar.selectbox(
    "🧭 Wind Direction at 3PM", 
    wind3pm_options,
    help="Select cardinal wind direction at 3PM (e.g., NE, SW)"
)
WindDir3pm = label_encoder_wind3pm.transform([wind3pm_name])[0]
st.sidebar.markdown(f"**Encoded WindDir3pm:** `{WindDir3pm}`")

# Rain Today
RainToday = st.sidebar.radio("Did it rain today?", ["No", "Yes"])
RainToday = 1 if RainToday == "Yes" else 0

# Current date
now = datetime.datetime.now()
year, month, day = now.year, now.month, now.day

# Create input DataFrame
input_data = pd.DataFrame([[
    MinTemp, Rainfall, Evaporation, Sunshine, WindGustSpeed,
    WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm,
    Pressure9am, Cloud9am, Cloud3pm, Location, WindGustDir,
    WindDir9am, WindDir3pm, RainToday, year, month, day
]], columns=[
    "MinTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed",
    "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
    "Pressure9am", "Cloud9am", "Cloud3pm", "Location", "WindGustDir",
    "WindDir9am", "WindDir3pm", "RainToday", "year", "month", "day"
])

# Predict
if st.sidebar.button("🚀 Predict Rain Tomorrow"):
    prediction = model.predict(input_data)[0]
    label = "🌧 Yes, it will rain." if prediction == 1 else "🌞 No, it won't rain."
    
    input_data["Prediction"] = label
    st.session_state.history = pd.concat([st.session_state.history, input_data], ignore_index=True)
    
    st.subheader(f"📡 Prediction: {label}")

# Show input and history
st.write("🔍 *Latest Prediction Input:*")
st.write(input_data)

st.write("📜 Prediction History")
st.write(st.session_state.history)

# Download button
csv = st.session_state.history.to_csv(index=False)
st.download_button("Download Prediction History", data=csv, file_name="rain_prediction_history.csv", mime="text/csv")