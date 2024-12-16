# Submission 2 : Proyek MLOps Depression Student

Nama: Abid Juliant Indraswara

Username dicoding:

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Depression Student Dataset](https://www.kaggle.com/datasets/ikynahidwin/depression-student-dataset) |
| Masalah | Performa belajar siswa umumnya dilihat melalui capaian nilai hasil yang didapatkan dari mata pelajaran yang diambil. Selain dari capaian nilai performa siswa juga dilihat secara sisi psikologi yang didukung data-data latar belakang siswa. Performa siswa yang baik dan buruk dapat diketahui melalui depresi tidaknya siswa. Depresi adalah gangguan suasana hati yang ditandai dengan perasaan sedih, putus asa, dan kehilangan minat atau kegembiraan dalam aktivitas yang biasanya menyenankan. Depresi pada siswa merupakan masalah yang semakin mendapat perhatian dalam pendidikan. Berbagai faktor, baik internal maupun eksternal, dapat mempengaruhi kesehatan mental seorang siswa. Di kalangan pelajar, depresi sering kali diakibatkan oleh tekanan akademik, masalah sosial, dan tantangan emosional yang datang dengan proses perkembangan diri di masa remaja. Sehingga pendekatan secara psikologi data latar belakang siswa dalam mengetahui siswa yang depresi mampu memberikan penanganan yang tepat. |
| Solusi machine learning | Sistem model machine learning dikembangkan untuk mengklasifikasikan siswa yang berisiko mengalami depresi. Data yang digunakan meliputi informasi tentang kesehatan mental siswa, hasil akademik, perilaku sosial, serta data demografis. Setiap data siswa akan dilabeli dengan dua kategori: mengalami depresi atau tidak mengalami depresi. Untuk mengolah data yang mengandung berbagai tipe informasi, model ini memanfaatkan kombinasi fitur numerik dan kategorikal. Fitur numerik seperti nilai ujian, usia, dan waktu tidur diproses langsung, sementara fitur kategorikal seperti jenis kelamin, status sosial, dan tingkat partisipasi dalam kegiatan ekstrakurikuler diubah menggunakan teknik One-Hot Encoding. Arsitektur model neural network digunakan untuk menangani data yang kompleks dan saling terkait. Lapisan Dense pada model ini menghubungkan seluruh fitur yang digabungkan, memungkinkan model belajar dari hubungan non-linear antara data numerik dan kategorikal. Lapisan-lapisan tersembunyi berturut-turut mengurangi dimensi data dan mengekstraksi fitur penting untuk mengklasifikasikan apakah seorang siswa berisiko mengalami depresi. Output model menggunakan fungsi aktivasi sigmoid, menghasilkan nilai probabilitas yang menunjukkan kemungkinan siswa tersebut mengalami depresi. Hasil dari model ini adalah kemampuan untuk mengklasifikasikan siswa ke dalam dua kategori: "depresi" atau "tidak depresi". Model ini memberikan dukungan kepada pihak sekolah atau orang tua untuk melakukan intervensi lebih cepat dan lebih tepat dalam mengatasi masalah kesehatan mental siswa. |
| Metode pengolahan | Data memiliki satu fitur yang digunakan dalam klasifikasi label negatif (Depresi) bernilai 0 dan positif (Non-Depresi) bernilai 1. Fitur kolom Depresi yang dimiliki dataset merupakan data yang berisi data siswa Depresi yang terdiri dari kolom kategorik dan kolom numerik. Data kategorik terdiri dari "Gender", "Sleep Duration", "Diatery Habbit, "Have you ever had suicidal thoughts ?" dan "Family History of Mental Illness". Data numerik terdiri dari kolom "Age", "Academic Pressure", "Study Satisfaction", "Study Hours" dan "Financial Stress". Terdapat juga kolom Label yaitu "Depression". Data dibagi ke dalam perbandingan 80% untuk training dan 20% untuk testing. Setelah itu proses transformasi data kategorik menjadi numerik tujuannya agar data dapat diolah oleh sistem dengan mudah. Sedangkan data numerik akan dilakukan transformasi untuk normalisasi diubah ke dalam skala 0 hingga 1 menggunakan one hot encode yang berfungsi untuk memudahkan sistem melakukan training model |
| Arsitektur model | Model ini dibangun menggunakan TensorFlow dan Keras untuk memproses data yang berisi fitur kategorikal dan numerik. Fitur kategorikal diubah menjadi representasi One-Hot Encoding, sementara fitur numerik diproses langsung sebagai input numerik. Semua input fitur, baik kategorikal maupun numerik, digabungkan menggunakan lapisan concatenate untuk memungkinkan pemrosesan informasi dari berbagai tipe fitur dalam satu jaringan. Model ini memiliki tiga lapisan tersembunyi berturut-turut dengan jumlah neuron masing-masing 256, 64, dan 16, menggunakan fungsi aktivasi ReLU untuk mempelajari pola non-linear dari data. Lapisan output terdiri dari satu neuron dengan fungsi aktivasi sigmoid, yang menghasilkan nilai antara 0 dan 1 untuk klasifikasi biner apakah seorang siswa mengalami depresi atau tidak. Model ini dikompilasi menggunakan optimizer Adam dengan learning rate 0.001, dan menggunakan fungsi kerugian binary_crossentropy serta metrik BinaryAccuracy untuk mengukur akurasi model. |
| Metrik evaluasi | Metrik evaluasi yang diterapkan dalam proyek ini mencakup AUC, Precision, Recall, ExampleCount, dan BinaryAccuracy. ExampleCount digunakan untuk menghitung jumlah contoh data yang telah dievaluasi, memberikan gambaran umum mengenai volume data yang digunakan selama proses evaluasi. BinaryCrossentropy berfungsi untuk mengukur tingkat kesalahan atau kerugian model dalam konteks klasifikasi biner, mencerminkan seberapa baik model dapat memisahkan dua kelas yang ada. BinaryAccuracy menunjukkan persentase klasifikasi yang benar dibandingkan dengan total prediksi, memberikan gambaran tentang akurasi keseluruhan model. Sementara itu, Precision mengukur seberapa tepat model dalam mendeteksi kelas positif (misalnya, kasus "siswa tidak depresi"), dan Recall menilai kemampuan model dalam menangkap seluruh kasus positif yang ada di dataset, menghindari ketidaktepatan dalam mendeteksi kelas tersebut. Seluruh metrik ini bekerja bersama untuk mengevaluasi performa model dalam mengklasifikasikan data, dengan ambang batas keputusan yang telah ditetapkan pada 0.5. Dengan konfigurasi evaluasi yang komprehensif ini, diharapkan model dapat memberikan prediksi yang akurat dan dapat diandalkan, khususnya dalam menentukan apakah siswa depresi atau tidak.|
| Performa model | Pelatihan model klasifikasi siswa yang depresi menunjukkan performa yang sangat baik setelah 5 epoch. Pada epoch terakhir, model berhasil mencapai akurasi pelatihan sebesar 100% dengan loss yang sangat rendah, yaitu 2.2202e-09, menunjukkan bahwa model sepenuhnya mempelajari pola dalam data pelatihan. Akurasi pada data validasi juga tetap tinggi, yaitu 94.19%, meskipun sedikit menurun dibandingkan dengan pelatihan. Loss validasi berada pada 0.2666, yang menunjukkan bahwa model mampu menggeneralisasi dengan baik pada data yang belum pernah dilihat sebelumnya. Secara keseluruhan, model menunjukkan kemampuan yang sangat baik dalam mengklasifikasikan siswa yang berisiko depresi dengan akurasi yang tinggi dan performa yang stabil pada data validasi. |
| Opsi deployment | Proyek yang dibuat disusun dalam sebuah pipeline machine learning di deploy ke platform as a service yaitu Railway menyediakan layanan gratis untuk melakukan deploy. |
| Web app | Tautan web app yang digunakan untuk mengakses model serving ada pada link berikut [depression-student-model](https://depression-student-prediction-production.up.railway.app/v1/models/depression-student-model/metadata)|
| Monitoring | Monitor model serving menggunakan layanan gratis open-source yaitu dengan Prometheus. Sistem monitoring ini mencatat setiap request yang masuk ke dalam sistem. Kemudian data akan dimonitor melalui status dari setiap request yang ada dan diterima. |