"""
    Author      : Abid Indraswara
    Date Create : 11 December 2024
    Modul Transform Data
    Fungsi :
    - Modul untuk transformasi data seperti 
      mengubah data kategori menjadi numerik dan
      mengubah data numerik menjadi skala tertentu -1 to 1 atau 0 to 1.
"""

# Import Library Reguler Expression
import re

# Import Library Tensorflow
import tensorflow as tf
import tensorflow_transform as tft

# List Variabel Fitur
# Fitur Kategori Dataset Depression Students
CATEGORICAL_FEATURES = {
    "Gender": 2,
    "Sleep Duration": 4,
    "Dietary Habits": 3,
    "Have you ever had suicidal thoughts ?": 2,
    "Family History of Mental Illness": 2
}

# Fitur Numerik Dataset Depression Students
NUMERICAL_FEATURES = [
    "Age",
    "Academic Pressure",
    "Study Satisfaction",
    "Study Hours",
    "Financial Stress"
]

# Label Klasifikasi Dataset Depression Students
LABEL_KEY = "Depression"

# Fungsi transform kategori dalam bentuk string ke numerik
def transformed_name(key):
    """Renaming transformed features"""
    key_transformed = re.sub(r'[^a-zA-Z0-9]', '', key)  # Menghapus semua karakter selain huruf dan angka
    return key_transformed + "_xf"

# Fungsi transform numerik menggunakan one hot
def convert_num_to_one_hot(label_tensor, num_labels=2):
    """Convert a label tensor into one-hot encoding."""
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])

# Fungsi preprocessing data
def preprocessing_fn(inputs):
    """Preprocess input features into transformed features."""

    outputs = {}

    # Proses fitur kategorikal lainnya
    for key in CATEGORICAL_FEATURES:
        dim = CATEGORICAL_FEATURES[key]

        # Menggunakan compute_and_apply_vocabulary untuk konversi kategori menjadi indeks
        data = inputs[key]
        int_value = tft.compute_and_apply_vocabulary(data, top_k=dim + 1)  # Dim + 1 karena termasuk semua kategori

        # Mengonversi nilai numerik ke one-hot encoding
        one_hot_values = convert_num_to_one_hot(int_value, num_labels=dim + 1)

        # Menyimpan hasil one-hot encoding di outputs
        outputs[transformed_name(key)] = one_hot_values

    # Memproses fitur numerik
    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])

    # Menyimpan hasil transformasi numerik untuk LABEL_KEY
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs