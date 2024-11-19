from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import joblib
import graphviz

# Path ke model yang sudah dilatih
model_path_stunting = 'decision_tree_model.pkl'  # Model untuk prediksi stunting
model_path_menu = 'menu_decision_tree_model.pkl'  # Model untuk rekomendasi menu

# Fungsi untuk menyimpan decision tree sebagai gambar
def simpan_decision_tree(model, feature_names, filename):
    # Ambil nama kelas secara otomatis
    class_names = [str(cls) for cls in model.classes_]
    
    # Buat file .dot untuk decision tree
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    # Konversi file .dot ke gambar dengan Graphviz
    graph = graphviz.Source(dot_data)
    graph.render(filename, format="png")  # Simpan gambar dalam format PNG
    print(f"Decision tree berhasil disimpan sebagai {filename}.png")


# 1. Visualisasi Decision Tree untuk Prediksi Stunting
print("Memuat model decision tree untuk stunting...")
dt_classifier = joblib.load(model_path_stunting)

# Fitur untuk stunting
feature_names_stunting = ['Tinggi Badan (cm)', 'Berat Badan (kg)', 'Umur Balita (bulan)', 'Jenis Kelamin', 'Tingkat Aktivitas']

# Simpan gambar decision tree
simpan_decision_tree(dt_classifier, feature_names_stunting, "decision_tree_stunting")


# 2. Visualisasi Decision Tree untuk Rekomendasi Menu
print("Memuat model decision tree untuk rekomendasi menu...")
dt_menu_classifier = joblib.load(model_path_menu)

# Fitur untuk menu
feature_names_menu = ['Tinggi Badan (cm)', 'Berat Badan (kg)', 'Umur Balita (bulan)', 'Jenis Kelamin', 'Tingkat Aktivitas']

# Simpan gambar decision tree
simpan_decision_tree(dt_menu_classifier, feature_names_menu, "decision_tree_menu")
