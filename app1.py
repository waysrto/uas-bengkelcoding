import streamlit as st
import pandas as pd
import joblib
import os

# Load the Random Forest model and scaler
random_forest_model = joblib.load('random_forest.pkl')  # Pastikan file model berada di direktori yang sama
scaler = joblib.load('scaler.pkl')  # Pastikan file scaler berada di direktori yang sama

# Fungsi untuk menyematkan file CSS
def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Memuat CSS
load_css("styles.css")

# Streamlit App Title
st.title("Prediksi Kualitas Air")
st.write("Aplikasi ini memprediksi apakah air dapat diminum (aman untuk dikonsumsi) berdasarkan parameter kualitas air.")

# Sidebar untuk input pengguna
st.sidebar.header("Fitur Input")
ph = st.sidebar.slider("Tingkat pH", 0.0, 14.0, 7.0)
hardness = st.sidebar.slider("Kekerasan (mg/L)", 0.0, 300.0, 100.0)
solids = st.sidebar.slider("Padatan Terlarut (mg/L)", 0.0, 50000.0, 20000.0)
chloramines = st.sidebar.slider("Kloramin (ppm)", 0.0, 12.0, 6.0)
sulfate = st.sidebar.slider("Sulfat (mg/L)", 0.0, 500.0, 200.0)
conductivity = st.sidebar.slider("Konduktivitas (uS/cm)", 0.0, 800.0, 400.0)
organic_carbon = st.sidebar.slider("Karbon Organik (ppm)", 0.0, 30.0, 15.0)
trihalomethanes = st.sidebar.slider("Trihalometana (ppb)", 0.0, 120.0, 60.0)
turbidity = st.sidebar.slider("Kekeruhan (NTU)", 0.0, 5.0, 2.5)

# Gabungkan input pengguna ke dalam DataFrame
input_data_indonesia = pd.DataFrame({
    'Tingkat pH': [ph],
    'Kekerasan (mg/L)': [hardness],
    'Padatan Terlarut (mg/L)': [solids],
    'Kloramin (ppm)': [chloramines],
    'Sulfat (mg/L)': [sulfate],
    'Konduktivitas (uS/cm)': [conductivity],
    'Karbon Organik (ppm)': [organic_carbon],
    'Trihalometana (ppb)': [trihalomethanes],
    'Kekeruhan (NTU)': [turbidity]
})

# Peta nama fitur bahasa Indonesia ke bahasa Inggris
input_data_english = input_data_indonesia.rename(columns={
    'Tingkat pH': 'ph',
    'Kekerasan (mg/L)': 'Hardness',
    'Padatan Terlarut (mg/L)': 'Solids',
    'Kloramin (ppm)': 'Chloramines',
    'Sulfat (mg/L)': 'Sulfate',
    'Konduktivitas (uS/cm)': 'Conductivity',
    'Karbon Organik (ppm)': 'Organic_carbon',
    'Trihalometana (ppb)': 'Trihalomethanes',
    'Kekeruhan (NTU)': 'Turbidity'
})

# Normalisasi input pengguna menggunakan scaler yang telah disimpan
scaled_input = scaler.transform(input_data_english)

# Membuat prediksi menggunakan model Random Forest
prediction = random_forest_model.predict(scaled_input)[0]
prediction_label = "Dapat Diminum (Aman untuk Dikonsumsi)" if prediction == 1 else "Tidak Dapat Diminum (Tidak Aman untuk Dikonsumsi)"

# Menampilkan hasil prediksi
st.write("### Fitur Input Pengguna")
st.write(input_data_indonesia)

st.write("### Prediksi")
st.write(f"**{prediction_label}**")

# Menampilkan probabilitas prediksi (jika ada)
if hasattr(random_forest_model, "predict_proba"):
    probabilities = random_forest_model.predict_proba(scaled_input)
    st.write("### Probabilitas Prediksi")
    st.write(probabilities)
