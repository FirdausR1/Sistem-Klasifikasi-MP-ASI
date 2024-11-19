from flask import Flask, request, jsonify, render_template
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import random


app = Flask(__name__)

mp_asi_data_path = 'DATASET/Databayi1.csv'
resep_makanan_data_path = 'DATASET/Data_Rekomendasi_Menu.xlsx'

mp_asi_df = pd.read_csv(mp_asi_data_path)
resep_makanan_df = pd.read_excel(resep_makanan_data_path)

resep_makanan_df['Kalori (kkal)'] = pd.to_numeric(resep_makanan_df['Kalori (kkal)'], errors='coerce')


def cek_usia(umur, rentang_usia):
    rentang = rentang_usia.split('-')
    if len(rentang) == 2:
        min_usia = int(rentang[0])
        max_usia = int(rentang[1])
        return min_usia <= umur <= max_usia
    return False

def rentang_usia_terdekat(umur, resep_df):
    usia_dataset = resep_df['Umur Balita (bulan)'].apply(lambda x: int(x.split('-')[0]))  
    perbedaan = abs(usia_dataset - umur) 
    indeks_terdekat = perbedaan.idxmin() 
    rentang_usia = resep_df.iloc[indeks_terdekat]['Umur Balita (bulan)']  
    return rentang_usia

def hitung_z_score(nilai, rata_rata, standar_deviasi):
    return (nilai - rata_rata) / standar_deviasi

def hitung_z_score_who(tinggi_badan, berat_badan, umur, jenis_kelamin):
    if jenis_kelamin == 0: 
        tinggi_rata_rata = 75 + 2 * (umur / 12)  
        berat_rata_rata = 10 + 0.5 * (umur / 12)  
    else: 
        tinggi_rata_rata = 73 + 1.8 * (umur / 12)  
        berat_rata_rata = 9.5 + 0.45 * (umur / 12)  
    
    std_tinggi = 3
    std_berat = 1.5
    
    z_score_tinggi = hitung_z_score(tinggi_badan, tinggi_rata_rata, std_tinggi)
    z_score_berat = hitung_z_score(berat_badan, berat_rata_rata, std_berat)
    
    return z_score_tinggi, z_score_berat

def hitung_bmr(berat_badan, tinggi_badan, umur, jenis_kelamin):
    if jenis_kelamin == 0: 
        return 88.362 + (13.397 * berat_badan) + (4.799 * tinggi_badan) - (5.677 * umur)
    else: 
        return 447.593 + (9.247 * berat_badan) + (3.098 * tinggi_badan) - (4.330 * umur)


def hitung_kebutuhan_kalori(umur, berat_badan, tinggi_badan, jenis_kelamin, tingkat_aktivitas):
    bmr = hitung_bmr(berat_badan, tinggi_badan, umur, jenis_kelamin)
    
    faktor_aktivitas = 1.2 if tingkat_aktivitas == 0 else 1.55 if tingkat_aktivitas == 1 else 1.725
    total_kebutuhan_kalori = bmr * faktor_aktivitas


    if 6 <= umur <= 8:
        persentase_mpasi = 0.3 
    elif 9 <= umur <= 11:
        persentase_mpasi = 0.5 
    elif 12 <= umur <= 23:
        persentase_mpasi = 0.7  
    else:
        persentase_mpasi = 1.0 

    
    kebutuhan_kalori_mpasi = total_kebutuhan_kalori * persentase_mpasi
    return kebutuhan_kalori_mpasi



def rekomendasi_menu(kebutuhan_kalori_mpasi, umur, resep_df, jumlah_rekomendasi=5):

    batas_bawah = kebutuhan_kalori_mpasi - 50
    batas_atas = kebutuhan_kalori_mpasi + 50
 
    rekomendasi = resep_df[
        (resep_df['Umur Balita (bulan)'].apply(lambda x: cek_usia(umur, str(x)))) &
        (resep_df['Kalori (kkal)'] >= batas_bawah) &
        (resep_df['Kalori (kkal)'] <= batas_atas)
    ]


    if rekomendasi.empty:
        rekomendasi = resep_df[resep_df['Umur Balita (bulan)'].apply(lambda x: cek_usia(umur, str(x)))]
        rekomendasi['Perbedaan Kalori'] = abs(rekomendasi['Kalori (kkal)'] - kebutuhan_kalori_mpasi)
        rekomendasi = rekomendasi.sort_values(by='Perbedaan Kalori').reset_index(drop=True)
    

    if not rekomendasi.empty:
        rekomendasi_menu = rekomendasi.sample(n=min(jumlah_rekomendasi, len(rekomendasi)))
    else:
        rekomendasi_menu = pd.DataFrame()
    
    return rekomendasi_menu[['Nama Makanan', 'Kalori (kkal)', 'Bahan', 'Cara Membuat', 'Porsi']]


label_encoder_jenis_kelamin = LabelEncoder()
mp_asi_df['Jenis Kelamin'] = label_encoder_jenis_kelamin.fit_transform(mp_asi_df['Jenis Kelamin'])

label_encoder_tingkat_aktivitas = LabelEncoder()
mp_asi_df['Tingkat Aktivitas'] = label_encoder_tingkat_aktivitas.fit_transform(mp_asi_df['Tingkat Aktivitas'])


X = mp_asi_df[['Tinggi Badan (cm)', 'Berat Badan (kg)', 'Umur Balita (bulan)', 'Jenis Kelamin', 'Tingkat Aktivitas']]
y = mp_asi_df['Stunting (Kemenkes)']

# Split data training dan testing 70:30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# SMOTE untuk mengatasi ketidakseimbangan data
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Pelatihan model untuk klasifikasi stunting
model_filename = 'decision_tree_model.pkl'
if os.path.exists(model_filename):
    dt_classifier = joblib.load(model_filename)
    print("Model loaded from disk.")
else:
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train_sm, y_train_sm)  # Train dengan data seimbang
    joblib.dump(dt_classifier, model_filename)
    print("Model trained and saved to disk.")

X_menu = mp_asi_df[['Tinggi Badan (cm)', 'Berat Badan (kg)', 'Umur Balita (bulan)', 'Jenis Kelamin', 'Tingkat Aktivitas']].iloc[:216]
y_menu = resep_makanan_df['Kategori Menu']  # Use the new 'Menu Category' column as the target
X_train_menu, X_test_menu, y_train_menu, y_test_menu = train_test_split(X_menu, y_menu, test_size=0.3, random_state=42)

# Terapkan SMOTE untuk rekomendasi menu
X_train_menu_sm, y_train_menu_sm = smote.fit_resample(X_train_menu, y_train_menu)

dt_menu_classifier = DecisionTreeClassifier()
dt_menu_classifier.fit(X_train_menu_sm, y_train_menu_sm)
# Save the model
menu_model_filename = 'menu_decision_tree_model.pkl'
joblib.dump(dt_menu_classifier, menu_model_filename)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/recommendation')
def recommendation():
    return render_template('index.html')

@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/predict', methods=['POST'])
def predict():
    umur = int(request.form['umur_bulan'])
    berat_badan = float(request.form['berat_badan'])
    tinggi_badan = float(request.form['tinggi_badan'])
    jenis_kelamin = int(request.form['jenis_kelamin'])
    tingkat_aktivitas = int(request.form['tingkat_aktivitas'])

    kebutuhan_kalori = hitung_kebutuhan_kalori(umur, berat_badan, tinggi_badan, jenis_kelamin, tingkat_aktivitas)

    

    input_prediksi = pd.DataFrame({
        'Tinggi Badan (cm)': [tinggi_badan],
        'Berat Badan (kg)': [berat_badan],
        'Umur Balita (bulan)': [umur],
        'Jenis Kelamin': [jenis_kelamin],
        'Tingkat Aktivitas': [tingkat_aktivitas]
    })
    hasil_prediksi = dt_classifier.predict(input_prediksi)
    status_stunting = "Stunting" if hasil_prediksi == 1 else "Normal"


    kategori_menu = dt_menu_classifier.predict(input_prediksi)[0]

   
    menu_rekomendasi = resep_makanan_df[resep_makanan_df['Kategori Menu'] == kategori_menu]

    if not menu_rekomendasi.empty:
        predicted_menu = menu_rekomendasi.sample(n=min(5, len(menu_rekomendasi))).to_dict(orient='records')
    else:
        predicted_menu = "Tidak ada menu yang sesuai untuk kategori yang diprediksi."


    return jsonify({
        'status_stunting': status_stunting,
        'kalori': round(kebutuhan_kalori),
        'kategori_menu': kategori_menu,
        'predicted_menu': predicted_menu,
      
    })

if __name__ == '__main__':
    app.run(debug=True)
