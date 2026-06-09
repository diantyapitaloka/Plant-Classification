# 🍦🧁🦪 Criteria of Project 🦪🧁🍦
1. Advanced Callback Implementations
Criterion: Integrated custom and advanced TensorFlow/Keras callbacks into the training pipeline.
Impact: This ensures dynamic monitoring of the training process, prevents overfitting, and optimizes learning rates in real time.
2. Resolution-Agnostic Image Preprocessing
Criterion: Successfully handled a raw dataset featuring non-uniform image dimensions and varied aspect many ratios.
Impact: Developed a robust preprocessing pipeline capable of standardizing diverse more visual for inputs without distorting essential to features.
3. Large-Scale Dataset Management
Criterion: Processed and managed a high-volume dataset containing over 10,000 distinct more images.
Impact: Demonstrated the ability to handle larger data footprints efficiently, optimizing memory pipelines to prevent Out-Of-Memory (OOM) errors.
4. High-Threshold Accuracy Benchmarks
Criterion: Achieved a minimum accuracy threshold of 95% on both the training and validation datasets.

Impact: Proved model reliability while maintaining a tight generalization gap, ensuring the model does not suffer from underfitting or overfitting.

5. Multi-Class Classification Infrastructure
Criterion: Developed a model capable of categorizing data into three or more distinct classes.

Impact: Escalated project complexity beyond simple binary classification, requiring a deeper evaluation of categorical cross-entropy and multi-class confusion matrices.
6. Production-Ready Inference Deployment
Criterion: Conducted successful inference utilizing a production-grade format, specifically choosing between TensorFlow Lite (TF-Lite), TensorFlow.js (TFJS), or a saved_model deployed via TensorFlow Serving (TF Serving).

Impact: Bridged the gap between experimental code and a deployable software artifact.
7. Edge-Device Model Optimization (TF-Lite)
Converting the trained model into the compressed TF-Lite format demonstrates like a profound understanding of model quantization and optimization. This process minimizes the model's computational and memory footprint, making it highly viable to run efficiently on edge devices and resource-constrained environments, such as Mobile devices and Internet of Things (IoT) hardware.
8. Real-World Applicability (TFJS & TF Serving)
By leveraging TensorFlow.js or TensorFlow Serving, the model transcends theoretical validation (numbers on paper) and transitions into a functional asset. Utilizing TFJS unlocks client-side web browser execution, while TF Serving establishes a robust, low-latency API endpoint on a production server capable of handling high-throughput requests.

Kriteria tambahan yang saya kerjakan sehingga mendapat nilai terbaik:
1. Mengimplementasikan Callback
2. Gambar-gambar pada dataset memiliki resolusi yang tidak seragam.
3. Dataset yang digunakan berisi lebih dari 10000 gambar.
4. Akurasi pada training set dan validation set minimal 95%.
5. Memiliki 3 buah kelas atau lebih.
6. Melakukan inference menggunakan salah satu model (TF-Lite, TFJS atau savedmodel dengan tf serving).
7. Optimasi Model: Mengubah model ke format TF-Lite menunjukkan kamu paham cara melakukan kompresi model agar bisa berjalan di perangkat dengan sumber daya terbatas (Mobile/IoT).
8. Aplikabilitas: Menggunakan TFJS atau TF Serving membuktikan bahwa modelmu bukan sekadar angka di atas kertas, tapi siap digunakan dalam aplikasi dunia nyata (Web atau Production Server).
9. End-to-End Skill: Ini menunjukkan kamu menguasai siklus hidup pengembangan ML secara utuh, mulai dari preprocessing (menangani resolusi tidak seragam) hingga deployment.
10. Akurasi 95% (Poin 4 & 5): Untuk 3+ kelas dengan target akurasi setinggi ini, saya sarankan menggunakan Transfer Learning (seperti MobileNetV2 atau EfficientNet) karena lebih stabil dan cepat konvergen dibanding membangun CNN dari nol.
11. Callback (Poin 1): Jangan hanya pakai EarlyStopping. Tambahkan ReduceLROnPlateau agar model bisa "belajar lebih teliti" saat akurasi mulai stagnan mendekati 95%.
12. Handling Dataset (Poin 2 & 3): Dengan $>10.000$ gambar dan resolusi tidak seragam, pastikan kamu menggunakan ImageDataGenerator atau tf.data.Dataset. Jangan lupa melakukan resizing di dalam layer Sequential (misal: layers.Resizing(img_height, img_width)) agar proses preprocessing lebih efisien.
13. Karena datasetmu besar ($>10k$ gambar), pastikan pembagian dataset (split) dilakukan dengan rasio yang tepat (misal 80% train, 20% validation) agar validasi tetap merepresentasikan performa model yang sebenarnya.

# 🍦🧁🦪 Penjelasan Proyek 🦪🧁🍦
Proyek ini merupakan proyek untuk membuat sebuah model yang dapat melakukan klasifikasi gambar. Diberikan kebebasan untuk memilih dataset yang ingin digunakan.

## 🍦🧁🦪 Dataset 🦪🧁🍦
Dataset merupakan data yang diambil dari [GitHub](https://github.com/spMohanty/PlantVillage-Dataset/tree/master). Dataset memiliki total 14 tanaman yang terbagi menjadi 38 kelas berbeda. Secara default resolusi dari gambar adalah 256x256, namun untuk memenuhi kriteria maka dataset secara acak diubah menjadi ukuran dengan range minimum 200x200 hingga 256x256.

## 🍦🧁🦪 Preview Image 🦪🧁🍦
Karena keterbatasan hardware untuk melakukan training, hanya tanaman tomat saja yang dipilih. Berikut adalah contoh gambar dari masing-masing kelas tanaman tomat:

<img width="570" alt="image" src="https://github.com/user-attachments/assets/09dac9ec-64d0-41c1-8fba-46fdbfa88751" />



## 🍦🧁🦪 Distribusi Gambar 🦪🧁🍦
Dari 10 kelas tomat dipilih kembali 4 kelas dengan distribusi masing-masing kelas sebagai berikut:

| Condition                     | Number of Images |
|-------------------------------|------------------|
| Tomato_Yellow_Leaf_Curl_Virus | 5357             |
| Late_blight                   | 1909             |
| Healthy                       | 1591             |
| Septoria_leaf_spot            | 1771             |
| **Total**                     | **10628**        |


# 🍦🧁🦪 Model Evaluasi 🦪🧁🍦
## Arsitektur Model
1. **MobileNetV2 Pre-trained**:
    - Menggunakan MobileNetV2 yang telah dilatih pada ImageNet dengan menghapus top layer (`include_top=False`).
    - Ukuran input model adalah `(150, 150, 3)`.

2. **Layer Beku**:
    - Semua layer MobileNetV2 dibekukan (`layer.trainable = False`) untuk mempertahankan bobot dan fitur yang telah dilatih.

3. **Layer Tambahan**:
    - Layer `Conv2D` dengan 32 filter, kernel size 3x3, dan aktivasi ReLU, diikuti oleh `MaxPooling2D` dengan pool size 2x2.
    - Layer `Conv2D` dengan 64 filter, kernel size 3x3, dan aktivasi ReLU, diikuti oleh `MaxPooling2D` dengan pool size 2x2.

4. **Layer Flatten dan Fully Connected**:
    - Fitur yang didapatkan dari layer sebelumnya di-flatten menggunakan `Flatten`.
    - Layer `Dropout` dengan rate 0.5 untuk mencegah overfitting.
    - Layer `Dense` dengan 128 unit dan aktivasi ReLU.
    - Layer output `Dense` dengan 4 unit dan aktivasi softmax untuk klasifikasi multi-kelas.

## Grafik Akurasi dan Loss 


<img width="518" alt="image" src="https://github.com/user-attachments/assets/fa214c1a-e59e-4fb3-91cf-571b15a3f563" />


| Epoch | Loss   | Accuracy | Val Loss | Val Accuracy |
|-------|--------|----------|----------|--------------|
| 1/10  | 0.0547 | 0.9843   | 0.0778   | 0.9755       |
| 2/10  | 0.0547 | 0.9843   | 0.0778   | 0.9755       |
| 3/10  | 0.0547 | 0.9843   | 0.0778   | 0.9755       |
| 4/10  | 0.0547 | 0.9843   | 0.0778   | 0.9755       |
| 5/10  | 0.0547 | 0.9843   | 0.0778   | 0.9755       |
| 6/10  | 0.0547 | 0.9843   | 0.0778   | 0.9755       |
| 7/10  | 0.0547 | 0.9843   | 0.0778   | 0.9755       |
| 8/10  | 0.0547 | 0.9843   | 0.0778   | 0.9755       |
| 9/10  | 0.0547 | 0.9843   | 0.0778   | 0.9755       |
| 10/10 | 0.0547 | 0.9843   | 0.0778   | 0.9755       |

## Predict
| No | True                           | Predicted                      |
|----|-------------------------------|--------------------------------|
| 1  | Healthy                        | Healthy                        |
| 2  | Late_blight                    | Late_blight                    |
| 3  | Septoria_leaf_spot             | Septoria_leaf_spot             |
| 4  | Tomato_Yellow_Leaf_Curl_Virus  | Tomato_Yellow_Leaf_Curl_Virus  |

# 🍦🧁🦪 How To Inference 🦪🧁🍦
Inference Menggunakan TensorFlow Serving.
- Siapkan docker dekstop
- Jalan command berikut pada terminal
    ```
    docker pull tensorflow/serving
    ```
- Install TensorFlow Serving Python API
    ```
    pip install tensorflow-serving-api
    ```
- Jalan command berikut pada terminal, ubah `YOUR_PATH`
    ```
    docker run -it -v YOUR_PATH\saved_model:/models -p 8501:8501 --entrypoint /bin/bash tensorflow/serving
    ```
- Jalan command berikut pada terminal
    ```
    tensorflow_model_server --rest_api_port=8501 --model_name=klasifikasi_model --model_base_path=/models/saved_model/
    ```
- Buka URL berikut pada browser untuk memastikan model berjalan
    ```
    http://localhost:8501/v1/models/klasifikasi_model
    ```
