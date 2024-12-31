import streamlit as st
import pandas as pd
import joblib

# Load the Random Forest model and scaler
random_forest_model = joblib.load('random_forest.pkl')  # Pastikan file model berada di direktori yang sama
scaler = joblib.load('scaler.pkl')  # Pastikan file scaler berada di direktori yang sama

# Streamlit App Title
st.title("Water Quality Prediction")
st.write("This app predicts whether water is potable (safe for drinking) based on quality parameters.")

# Sidebar for user input
st.sidebar.header("Input Features")
ph = st.sidebar.slider("pH Level", 0.0, 14.0, 7.0)
hardness = st.sidebar.slider("Hardness", 0.0, 300.0, 100.0)
solids = st.sidebar.slider("Solids (mg/L)", 0.0, 50000.0, 20000.0)
chloramines = st.sidebar.slider("Chloramines (ppm)", 0.0, 12.0, 6.0)
sulfate = st.sidebar.slider("Sulfate (mg/L)", 0.0, 500.0, 200.0)
conductivity = st.sidebar.slider("Conductivity (uS/cm)", 0.0, 800.0, 400.0)
organic_carbon = st.sidebar.slider("Organic Carbon (ppm)", 0.0, 30.0, 15.0)
trihalomethanes = st.sidebar.slider("Trihalomethanes (ppb)", 0.0, 120.0, 60.0)
turbidity = st.sidebar.slider("Turbidity (NTU)", 0.0, 5.0, 2.5)

# Combine user input into a single row DataFrame
input_data = pd.DataFrame({
    'ph': [ph],
    'Hardness': [hardness],
    'Solids': [solids],
    'Chloramines': [chloramines],
    'Sulfate': [sulfate],
    'Conductivity': [conductivity],
    'Organic_carbon': [organic_carbon],
    'Trihalomethanes': [trihalomethanes],
    'Turbidity': [turbidity]
})

# Normalize user input using the saved scaler
scaled_input = scaler.transform(input_data)

# Make predictions using the Random Forest model
prediction = random_forest_model.predict(scaled_input)[0]
prediction_label = "Potable (Safe to Drink)" if prediction == 1 else "Not Potable (Unsafe to Drink)"

# Display the results
st.write("### User Input Features")
st.write(input_data)

st.write("### Prediction")
st.write(f"**{prediction_label}**")

# Optional: Show prediction probabilities
if hasattr(random_forest_model, "predict_proba"):
    probabilities = random_forest_model.predict_proba(scaled_input)
    st.write("### Prediction Probabilities")
    st.write(probabilities)
