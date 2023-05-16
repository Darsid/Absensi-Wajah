import cv2
import numpy as np
import csv
import fnmatch
import os
import datetime

# Inisialisasi CascadeClassifier untuk deteksi wajah
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Daftar nama orang yang terdaftar
nama_orang = ['DapaArhama']

# Buka file CSV untuk mencatat absensi
with open('absensi.csv', mode='a', newline='') as file_csv:
    csv_writer = csv.writer(file_csv)
    csv_writer.writerow(['Nama', 'Waktu Absen'])

    # Loop melalui setiap nama orang yang terdaftar
    for nama in nama_orang:
        # Loop melalui file dalam direktori saat ini
        for file in os.listdir():
            # Cocokkan nama file gambar dengan nama orang yang terdaftar
            if fnmatch.fnmatch(file, f"*{nama}*.jpg"):
                # Baca gambar
                img = cv2.imread(file)

                # Ubah gambar menjadi grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Deteksi wajah pada gambar
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                # Loop melalui setiap wajah yang terdeteksi
                for (x, y, w, h) in faces:
                    # Gambar rectangle di sekitar wajah
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

                    # Dapatkan waktu absen
                    waktu_absen = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Tulis nama dan waktu absen ke file CSV
                    csv_writer.writerow([nama, waktu_absen])

                    # Tampilkan nama di sekitar rectangle
                    cv2.putText(img, nama, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)



                # Tampilkan gambar dengan wajah yang terdeteksi
                cv2.imshow('Deteksi Wajah', img)
                cv2.waitKey(0)

# Tutup window
cv2.destroyAllWindows()
