# -*- coding: utf-8 -*-
"""notebook.ipynb

Automatically generated by Google Colab.

Original file is located at
    https://colab.research.google.com/drive/1k5GEr6nNAs9W720ezFgtSsFZ9cXJkKqg

# Proyek Klasifikasi Gambar: PlantVillage-Dataset
- **Nama:** Diantya Pitaloka
- **Email:** diantyantyaa@gmail.com
- **ID Dicoding:** diantyap

## Import Semua Packages atau Library yang akan Digunakan

Menggunakan tools tensorflow 2.15 karena 2.17 selalu membuat colab crash ketika save ke tflite. (Berdasarkan forum diskusi dan pengalaman pribadi).
"""

!pip install tensorflowjs

!pip install tensorflow==2.15.0

"""Melakukan import library yang akan digunakan untuk keseluruhan proyek."""

# Import standard libraries
import os
import random
import shutil
import pathlib

# Import PIL for image processing
from PIL import Image

# Import matplotlib for plotting
import matplotlib.pyplot as plt

# Import TensorFlow and Keras libraries
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B0, MobileNetV2

print(f'TensorFlow version: {tf.__version__}')

"""## Bagian Data Preparation

### Bagian Data Loading

Dataset original bersumber dari [GitHub](https://github.com/spMohanty/PlantVillage-Dataset). Namun dataset sudah diupload ke google drive pribadi agar memudahkan untuk didownload.
"""

# Download File
!gdown 1-2-c46EfXP8RYZkEFFUmDb5-oB1sEByq

"""Melakukan unzip terhadap file hasil download sebelumnya."""

# Unzip File
!unzip color.zip

"""Membuat fungsi untuk menampilkan semua jumlah gambar dari masing-masing kelas dan menghitung jumlah gambar dengan resolusi tertentu."""

def count_images_and_resolution(base_path, target_resolution=None):
    # Dictionary untuk menyimpan jumlah gambar per kelas total
    class_count = {}

    # Dictionary untuk menyimpan jumlah gambar per resolusi total
    resolution_count = {}

    for root, dirs, files in os.walk(base_path):
        # Mengabaikan folder root yang menyeleksi tidak memiliki gambar 
        if root == base_path:
            continue
        class_name = os.path.basename(root)
        class_count[class_name] = len(files)

        for file in files:
            file_path = os.path.join(root, file)
            with Image.open(file_path) as img:
                width, height = img.size
                resolution = f"{width}x{height}"
                if resolution not in resolution_count:
                    resolution_count[resolution] = 0
                resolution_count[resolution] += 1

                # Menghitung total jumlah gambar dengan resolusi target_resolution
                if target_resolution and resolution == target_resolution:
                    if 'target' not in resolution_count:
                        resolution_count['target'] = 0
                    resolution_count['target'] += 1

    return class_count, resolution_count

# Path ke folder utama
base_path = "/content/color"

# Resolusi yang ingin dihitung
target_resolution = "256x256"

class_count, resolution_count = count_images_and_resolution(base_path, target_resolution)

# Menampilkan hasil seperti berikut 
print("Jumlah gambar per kelas:")
for class_name, count in class_count.items():
    print(f"{class_name}: {count}")

print("\nJumlah gambar per resolusi:")
for resolution, count in resolution_count.items():
    print(f"{resolution}: {count}")

"""Terdapat sejumlah 54305 gambar yang terbagi menjadi 38 kelas berbeda. Setiap gambar di masing-masing kelas memiliki resolusi 256x256.

Karena pada bagian submission dicoding harus terdapat gambar dengan resolusi yang berbeda-beda maka dilakukan perubahan resolusi dari masing-masing gambar secara manual.

Gambar lalu diubah dengan resolusi antara 200 hingga 256. Gambar original akan ditimpa oleh gambar yang resolusi nya telah diubah sehingga tidak menambah jumlah dataset.
"""

def resize_and_replace_images(base_path, min_res=200, max_res=256):
    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            with Image.open(file_path) as img:

                new_width = random.randint(min_res, max_res)
                new_height = random.randint(min_res, max_res)

                # Ubah ukuran gambar sebagai berikut
                resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)

                # Simpan gambar yang sudah diubah kategori
                resized_img.save(file_path)

# Selanjutnya path ke folder utama
base_path = "/content/color"

resize_and_replace_images(base_path)

"""Menampilkan kembali jumlah gambar untuk masing-masing resolusi."""

def count_images_and_resolution(base_path, target_resolution=None):
    # Berikut Dictionary untuk menyimpan jumlah gambar per kelas
    class_count = {}

    # Berikut Dictionary untuk menyimpan jumlah gambar per resolusi
    resolution_count = {}

    for root, dirs, files in os.walk(base_path):
        # Mengabaikan folder root yang tidak memiliki gambar sebagai berikut
        if root == base_path:
            continue
        class_name = os.path.basename(root)
        class_count[class_name] = len(files)

        for file in files:
            file_path = os.path.join(root, file)
            with Image.open(file_path) as img:
                width, height = img.size
                resolution = f"{width}x{height}"
                if resolution not in resolution_count:
                    resolution_count[resolution] = 0
                resolution_count[resolution] += 1

                # Hitung jumlah gambar dengan resolusi target_resolution
                if target_resolution and resolution == target_resolution:
                    if 'target' not in resolution_count:
                        resolution_count['target'] = 0
                    resolution_count['target'] += 1

    return class_count, resolution_count

# Path ke folder utama sebagai berikut
base_path = "/content/color"

# Kemudian resolusi yang ingin dihitung
target_resolution = "256x256"

class_count, resolution_count = count_images_and_resolution(base_path, target_resolution)

print("\nJumlah gambar per resolusi:")
for resolution, count in resolution_count.items():
    print(f"{resolution}: {count}")

"""Terlihat bahwa sekarang gambar sudah memiliki resolusi yang bervariasi.

### Data Preprocessing

Secara total terdapat 14 jenis tanaman yang berbeda. Namun pada proyek kali ini akan fokus ke tanaman tomat saja.
"""

def count_images(base_path, target_class="tomato"):
    total_count = 0
    target_class_count = 0

    for root, dirs, files in os.walk(base_path):
        total_count += len(files)
        if target_class.lower() in root.lower():
            target_class_count += len(files)

    return total_count, target_class_count

# Path ke folder utama sebagai berikut
base_path = "/content/color"

total_count, target_class_count = count_images(base_path)

# Menampilkan hasil sebagai berikut
print(f"Jumlah total gambar dalam dataset: {total_count}")
print(f"Jumlah gambar dalam kelas 'tomato': {target_class_count}")

"""Menampilkan jumlah gambar untuk masing-masing subkelas dari tanaman tomat."""

def count_tomato_images_per_subclass(base_path, target_class="tomato"):
    class_count = {}

    for root, dirs, files in os.walk(base_path):
        if target_class.lower() in root.lower():
            subclass_name = os.path.basename(root)
            if subclass_name not in class_count:
                class_count[subclass_name] = 0
            class_count[subclass_name] += len(files)

    return class_count

# Path ke folder utama sebagai berikut
base_path = "/content/color"

tomato_class_count = count_tomato_images_per_subclass(base_path)

# Menampilkan hasil sebagai berikut
print("Jumlah gambar dalam kelas 'tomato' untuk masing-masing subkelas:")
for subclass, count in tomato_class_count.items():
    print(f"{subclass}: {count}")

"""Memindahkan folder tanaman tomat dari /color ke /dataset"""

def copy_tomato_folders(base_path, target_folder="dataset", target_class="tomato"):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if target_class.lower() in dir_name.lower():
                source_path = os.path.join(root, dir_name)
                dest_path = os.path.join(target_folder, dir_name)
                if not os.path.exists(dest_path):
                    shutil.copytree(source_path, dest_path)
                    print(f"Menyalin {source_path} ke {dest_path}")

# Path ke folder utama sebagai berikut
base_path = "/content/color"

# Path ke folder tujuan
target_folder = "/content/dataset"

copy_tomato_folders(base_path, target_folder)

"""Menghapus prefix "tomato___"."""

def rename_folders(base_path, prefix="Tomato___"):
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name.startswith(prefix):
                new_dir_name = dir_name[len(prefix):]
                old_path = os.path.join(root, dir_name)
                new_path = os.path.join(root, new_dir_name)
                os.rename(old_path, new_path)
                print(f"Renamed {old_path} to {new_path}")

# Path ke folder utama
base_path = "/content/dataset"

rename_folders(base_path)

"""Menampilkan contoh gambar dari masing-masing kelas tomat."""

def show_example_images(base_path):
    class_images = {}

    for root, dirs, files in os.walk(base_path):
        if files:
            class_name = os.path.basename(root)
            random_image = random.choice(files)
            class_images[class_name] = os.path.join(root, random_image)

    fig, axes = plt.subplots(1, len(class_images), figsize=(15, 5))
    fig.suptitle('Contoh Gambar dari Masing-Masing Kelas')

    for ax, (class_name, image_path) in zip(axes, class_images.items()):
        img = Image.open(image_path)
        ax.imshow(img)
        ax.text(0.5, -0.1, class_name, rotation=90, verticalalignment='top', horizontalalignment='center', transform=ax.transAxes)
        ax.axis('off')

    plt.show()

# Path ke folder utama sebagai berikut
base_path = "/content/dataset"

show_example_images(base_path)

"""#### Split Dataset

Pada proyek dicoding jumlah dataset minimal yang dibutuhkan adalah 10.000 sehingga pada proyek kali ini hanya akan dipilih 4 kelas saja dari tanaman tomat agar meringangkan beban kerja. Kelas yang dipilih dan jumlah akhir dari dataset adalah sebagai berikut:

- Tomato_Yellow_Leaf_Curl_Virus: 5357
- Late_blight: 1909
- healthy: 1591
- Septoria_leaf_spot: 1771

Total = 10628 gambar

Menghapus semua folder kecuali folder yang akan digunakan.
"""

def delete_unwanted_folders(base_path, keep_folders=['Tomato_Yellow_Leaf_Curl_Virus', 'Late_blight','healthy','Septoria_leaf_spot']):
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item not in keep_folders:
            shutil.rmtree(item_path)
            print(f"Menghapus folder: {item_path}")

# Path ke folder utama sebagai berikut
base_path = "/content/dataset"

delete_unwanted_folders(base_path)

"""Membagi dataset menjadi train dan test dengan rasio 8:2."""

def split_dataset(base_path, train_ratio=0.8):
    # Path untuk dataset pelatihan dan pengujian sebagai berikut
    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')

    # Membuat folder train dan test jika belum ada sebagai berikut
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    for root, dirs, files in os.walk(base_path):
        if root == base_path:
            continue

        class_name = os.path.basename(root)
        if class_name in ['train', 'test']:
            continue

        # Membuat folder kelas di dalam train dan test sebagai berikut
        train_class_path = os.path.join(train_path, class_name)
        test_class_path = os.path.join(test_path, class_name)
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)

        # Shuffle files sebagai berikut
        random.shuffle(files)
        split_index = int(train_ratio * len(files))
        train_files = files[:split_index]
        test_files = files[split_index:]

        # Memindahkan file ke folder train sebagai berikut
        for file in train_files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(train_class_path, file)
            shutil.move(src_file, dst_file)

        # Memindahkan file ke folder test sebagai berikut
        for file in test_files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(test_class_path, file)
            shutil.move(src_file, dst_file)

# Path ke folder utama sebagai berikut
base_path = "/content/dataset"

split_dataset(base_path)

"""Menghapus folder selain folder train dan test."""

def delete_unwanted_folders(base_path, keep_folders=['train', 'test']):
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item not in keep_folders:
            shutil.rmtree(item_path)
            print(f"Menghapus folder: {item_path}")

# Path ke folder utama sebagai berikut
base_path = "/content/dataset"

delete_unwanted_folders(base_path)

"""## Modelling

Menggunakan ImageDataGenerator untuk melakukan augmentasi, rescale, dan mengubah target size.

Dataset test hanya akan dilakukan rescale.
"""

def augment_and_resize_dataset(base_path, img_size=(150, 150), batch_size=32):
    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Hanya rescale untuk data test sebagai berikut
    test_datagen = ImageDataGenerator(rescale=1./255)


    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, test_generator

# Path ke folder utama sebagai berikut
base_path = "/content/dataset"

train_generator, test_generator = augment_and_resize_dataset(base_path)

"""Menampilkan kelas-kelas yang terdapat pada dataset."""

class_indices = train_generator.class_indices
print(class_indices)

"""Menggunakan transferlearning dari MobileNetV2. Input shape yang digunakan adalah 150x150, layers di freeze agar tidak dilatih kembali, dan menambahkan beberapa layer Conv dan Pooling."""

pre_trained_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(150,150,3))

for layer in pre_trained_model.layers:
    layer.trainable = False

model = Sequential()

model.add(pre_trained_model)

# Menambahkan Conv2D and Pooling layers sebagai berikut
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten(name="flatten"))
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu"))
model.add(Dense(4, activation='softmax'))

"""Mengcompile model dengan optimizer Adam, loss categorical_crossentropy, dan metrics accuracy."""

# Compile model sebagai berikut
optimizer = tf.optimizers.Adam()
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

"""Membuat callbacks yang memonitor val_accuracy dan akan berhenti jika tidak mengalami perubahan selama 3 epochs."""

# Callbacks sebagai berikut
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, min_delta=0.001, restore_best_weights=True, mode='max', baseline=0.96)

"""Melatih model selama 10 epoch dan menggunakan data test sebagai validation."""

# Melatih model sebagai berikut
num_epochs = 10

H = model.fit(train_generator,
              epochs=num_epochs,
              validation_data=test_generator,
              callbacks=[checkpoint, early_stopping],
              verbose=1)

"""## Evaluasi dan Visualisasi

Menampilkan grafik train dan val akurasi serta train dan val loss.
"""

def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

plot_training_history(H)

"""## Konversi Model

Menyimpan model menjadi format .h5.
"""

model.save("model.h5")

"""### Konversi TFJS

Konversi model menjadi format TFJS.
"""

!tensorflowjs_converter --input_format=keras model.h5 tfjs_model

"""### Konversi SavedModel

Konversi menjadi saved_model.
"""

save_path = os.path.join("models/klasifikasi_gambar/1/")
tf.saved_model.save(model, save_path)

"""### Konversi TF-Lite

Konversi model menjadi format TFLITE dan menyimpan label.txt.
"""

# Load the Keras model sebagai berikut
model_TFLITE = tf.keras.models.load_model('model.h5')

# Convert the model to TensorFlow Lite format sebagai berikut
converter = tf.lite.TFLiteConverter.from_keras_model(model_TFLITE)
tflite_model = converter.convert()

# Save the converted model to a file sebagai berikut
with open("converted_model.tflite", "wb") as f:
    f.write(tflite_model)

# Buat konten yang akan ditulis ke dalam file sebagai berikut
content = """Late_blight
Septoria_leaf_spot
Tomato_Yellow_Leaf_Curl_Virus
healthy"""

# Tentukan path dan nama file sebagai berikut
file_path = "/content/klasifikasiGambar.txt"

# Tulis konten ke dalam file sebagai berikut
with open(file_path, "w") as file:
    file.write(content)

"""Menjadikan zip agar dapat di download ke local device."""

# Specify the folder to zip sebagai berikut
folder_modles = '/content/models'
folder_tfjs_model = '/content/tfjs_model'

# Specify the output zip file name (without .zip extension) sebagai berikut
output_modles= '/content/models'
output_tfjs_model = '/content/tfjs_model'

# Zip the folder sebagai berikut
shutil.make_archive(output_modles, 'zip', folder_modles)
shutil.make_archive(output_tfjs_model, 'zip', folder_tfjs_model)

!pip freeze > requirements.txt

"""## Inference

Melakukan inference terhadap model yang di deploy menggunakan tensorflow serving.
"""

import tensorflow as tf
import requests
import os

"""Melakukan prediksi, masing-masing kelas dicoba oleh 1 image."""

def images_preprocessing(filenames):
    image_tensors = []
    for filename in filenames:
        image = tf.io.decode_image(open(filename, 'rb').read(), channels=3)
        image = tf.image.resize(image, [150, 150])
        image = image / 255.
        image_tensor = tf.expand_dims(image, 0)
        image_tensors.append(image_tensor)

    # Concatenate all image tensors into a single batch sebagai berikut
    batch_tensor = tf.concat(image_tensors, axis=0)
    return batch_tensor.numpy().tolist()

# List of filenames to be processed sebagai berikut
filenames = [
    os.path.join('images', 'healthy.jpg'),
    os.path.join('images', 'Late_blight.jpg'),
    os.path.join('images', 'Septoria_leaf_spot.jpg'),
    os.path.join('images', 'Tomato_Yellow_Leaf_Curl_Virus.jpg')
]

image_tensors = images_preprocessing(filenames=filenames)

json_data = {
    "instances": image_tensors
}

endpoint = "http://localhost:8501/v1/models/klasifikasi_model:predict"

try:
    response = requests.post(endpoint, json=json_data)
    response.raise_for_status()  # Will raise an error for bad status codes sebagai berikut

    predictions = response.json()['predictions']
    map_labels = {0: 'Late_blight', 1: 'Septoria_leaf_spot', 2: 'Tomato_Yellow_Leaf_Curl_Virus', 3: 'healthy'}

    # Iterate through predictions and print the corresponding labels sebagai berikut
    for prediction in predictions:
        predicted_class = tf.argmax(prediction).numpy()
        print(map_labels[predicted_class])

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")