import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os

# Set konfigurasi awal untuk visualisasi
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# Fungsi untuk memuat dataset asli dari file .csv
@st.cache_data
def load_data(file_path="diabetes.csv"):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat dataset: {e}")
        return None

# Fungsi untuk memuat model dari file .pkl
def load_model_from_file(file_path):
    try:
        model = joblib.load(file_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocessing data untuk prediksi
def preprocess_data_for_prediction(df):
    feature_cols = ['Glucose', 'Insulin', 'BMI']
    cols_to_replace = ['Glucose', 'Insulin', 'BMI']
    df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)
    df[cols_to_replace] = df[cols_to_replace].fillna(df[cols_to_replace].median())
    return df[feature_cols]

# Fungsi untuk melakukan prediksi
def predict_diabetes(model, data):
    try:
        predictions = model.predict(data)
        return predictions
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Fungsi untuk menghitung matriks konfusi dan evaluasi
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return cm, report

# Load dataset asli
df = load_data("diabetes.csv")
if df is None:
    st.stop()

# Preprocessing untuk analisis
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_replace] = df[cols_to_replace].replace(0, pd.NA)
df_cleaned = df.dropna().drop_duplicates()

# Set title for the dashboard
st.title("🩺 Dashboard Analisis Diabetes (Pima Indians)")
st.write("Dashboard ini menampilkan analisis data diabetes menggunakan dataset asli Pima Indians dengan visualisasi dan prediksi berbasis machine learning.")

# Tab navigasi
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Faktor Kesehatan", "📊 Pola Demografis", "🔮 Prediksi Diabetes", "📉 Evaluasi Model", "📚 Kesimpulan"])

# Sidebar untuk informasi tambahan
st.sidebar.title("📋 Informasi Tambahan")
st.sidebar.subheader("Statistik Ringkas")
avg_glucose = df_cleaned['Glucose'].mean()
st.sidebar.write(f"*Rata-rata Glukosa*: {avg_glucose:.2f}")
avg_bmi = df_cleaned['BMI'].mean()
st.sidebar.write(f"*Rata-rata BMI*: {avg_bmi:.2f}")
avg_age = df_cleaned['Age'].mean()
st.sidebar.write(f"*Rata-rata Umur*: {avg_age:.2f}")

if st.sidebar.button("Lihat Total Kasus Diabetes"):
    total_diabetes = df_cleaned['Outcome'].sum()
    st.sidebar.write(f"*Total Kasus Diabetes*: {total_diabetes}")

st.sidebar.subheader("Tentang Dataset")
st.sidebar.write("Dataset ini berisi data kesehatan wanita Pima Indian dari file `diabetes.csv`, termasuk faktor seperti glukosa, insulin, dan BMI untuk analisis diabetes.")

st.sidebar.subheader("ℹ Tentang Kelompok")
st.sidebar.write("*Anggota*:")
st.sidebar.write("- Damianus Christopher Samosir (G1A022028)")
st.sidebar.write("- Reksi Hendra Pratama (G1022032)")
st.sidebar.write("- Yuda Reyvandra Herman (G1A022072)")
st.sidebar.markdown("<p style='text-align: center; color: #A9A9A9;'>© 2025 Diabetes Analytics</p>", unsafe_allow_html=True)

# Tab 1: Faktor Kesehatan
with tab1:
    st.subheader("📈 Pengaruh Faktor Kesehatan terhadap Diabetes")
    st.write("Analisis faktor kesehatan berdasarkan dataset asli.")

    outcome_filter = st.selectbox(
        "Filter berdasarkan status diabetes:",
        ["Semua", "Tidak Diabetes (0)", "Diabetes (1)"],
        key="outcome_filter"
    )
    if outcome_filter != "Semua":
        outcome_value = int(outcome_filter.split("(")[1][0])
        filtered_df = df_cleaned[df_cleaned['Outcome'] == outcome_value].copy()
    else:
        filtered_df = df_cleaned.copy()

    st.write("*Hubungan Glukosa dan BMI terhadap Diabetes*")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=filtered_df, x='Glucose', y='BMI', hue='Outcome', size='Age', palette='deep', ax=ax1)
    ax1.set_title('Glucose vs BMI dengan Status Diabetes', fontsize=14)
    st.pyplot(fig1)

    st.write("*Distribusi Glukosa (Histogram)*")
    fig2, ax2 = plt.subplots()
    sns.histplot(data=filtered_df, x='Glucose', hue='Outcome', kde=True, palette='Set2', ax=ax2)
    ax2.set_title('Distribusi Kadar Glukosa', fontsize=14)
    st.pyplot(fig2)

    st.write("*Korelasi Semua Faktor*")
    correlation = filtered_df.corr()
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', ax=ax3)
    ax3.set_title('Matriks Korelasi', fontsize=14)
    st.pyplot(fig3)

# Tab 2: Pola Demografis
with tab2:
    st.subheader("📊 Pola Demografis dan Diabetes")
    st.write("Distribusi umur dan hubungan demografis dengan diabetes.")

    st.write("*Distribusi Umur (Histogram)*")
    fig4, ax4 = plt.subplots()
    sns.histplot(data=filtered_df, x='Age', hue='Outcome', kde=True, palette='Set3', ax=ax4)
    ax4.set_title('Distribusi Umur', fontsize=14)
    st.pyplot(fig4)

    st.write("*Umur dan Jumlah Kehamilan*")
    fig5, ax5 = plt.subplots()
    sns.boxplot(data=filtered_df, x='Outcome', y='Age', hue='Pregnancies', palette='Set2', ax=ax5)
    ax5.set_title('Umur dan Kehamilan Berdasarkan Diabetes', fontsize=14)
    st.pyplot(fig5)

# Tab 3: Prediksi Diabetes
with tab3:
    st.subheader("🔮 Prediksi Diabetes dengan Model Klasifikasi")
    st.write("Masukkan data pasien untuk prediksi diabetes.")

    model_choice = st.selectbox("Pilih Algoritma:", ["Random Forest", "Naive Bayes", "Decision Tree"])
    model_paths = {
        "Random Forest": "model_rf_diabetes.pkl",
        "Naive Bayes": "model_nb_diabetes.pkl",
        "Decision Tree": "model_dt_diabetes.pkl"
    }

    selected_model_path = model_paths.get(model_choice)
    selected_model = load_model_from_file(selected_model_path)

    if selected_model is None:
        st.warning(f"Model {model_choice} belum tersedia. Melatih model baru...")
        X = df_cleaned[['Glucose', 'Insulin', 'BMI']]
        y = df_cleaned['Outcome']
        X = X.replace(0, np.nan).fillna(X.median())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        if model_choice == "Random Forest":
            selected_model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_choice == "Naive Bayes":
            selected_model = GaussianNB()
        elif model_choice == "Decision Tree":
            selected_model = DecisionTreeClassifier(random_state=42)

        selected_model.fit(X_train, y_train)
        joblib.dump(selected_model, selected_model_path)
        st.success(f"Model {model_choice} berhasil dilatih dan disimpan.")

    col1, col2 = st.columns(2)
    with col1:
        glucose = st.number_input("Kadar Glukosa (mg/dL)", min_value=0.0, value=0.0)
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0.0, value=0.0)
    with col2:
        bmi = st.number_input("BMI", min_value=0.0, value=0.0)

    if st.button("Prediksi"):
        input_data = pd.DataFrame({'Glucose': [glucose], 'Insulin': [insulin], 'BMI': [bmi]})
        processed_input = preprocess_data_for_prediction(input_data)
        prediction = predict_diabetes(selected_model, processed_input)
        if prediction is not None:
            label_mapping = {0: "Normal", 1: "Diabetes"}
            result = label_mapping.get(prediction[0], "Tidak Diketahui")
            st.success(f"Hasil Prediksi dengan {model_choice}: *{result}*")
            if hasattr(selected_model, 'predict_proba'):
                probabilities = selected_model.predict_proba(processed_input)
                st.write(f"Probabilitas: Normal: {probabilities[0][0]:.2f}, Diabetes: {probabilities[0][1]:.2f}")

# Tab 4: Evaluasi Model
with tab5:
    st.subheader("📉 Evaluasi Model Klasifikasi")
    st.write("Matriks konfusi dan laporan evaluasi untuk masing-masing algoritma.")

    X = df_cleaned[['Glucose', 'Insulin', 'BMI']].replace(0, np.nan).fillna(df_cleaned[['Glucose', 'Insulin', 'BMI']].median())
    y = df_cleaned['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        cm, report = evaluate_model(model, X_test, y_test)

        st.write(f"### {name}")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Matriks Konfusi - {name}')
        ax.set_xlabel('Prediksi')
        ax.set_ylabel('Aktual')
        st.pyplot(fig)

        st.write(f"*Laporan Evaluasi - {name}:*")
        st.write(pd.DataFrame(report).T)

# Tab 5: Kesimpulan
with tab4:
    st.subheader("📚 Kesimpulan Analisis")
    st.write("Temuan utama dari analisis data diabetes:")
    st.write("""
    - *Faktor Kesehatan*: Glukosa dan BMI memiliki korelasi kuat dengan diabetes.
    - *Pola Demografis*: Usia di atas 30 tahun dan jumlah kehamilan tinggi meningkatkan risiko.
    - *Evaluasi Model*: Random Forest menunjukkan performa lebih baik dibandingkan Naive Bayes dan Decision Tree berdasarkan metrik evaluasi.
    - *Rekomendasi*: Fokus pada pemantauan glukosa, BMI, dan edukasi gaya hidup sehat.
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #A9A9A9;'>© 2025 Diabetes Analytics - UTS Penambangan Data</p>", unsafe_allow_html=True)